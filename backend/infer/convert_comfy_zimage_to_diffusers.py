"""Convertit un checkpoint Z-Image au format ComfyUI (.safetensors single-file)
vers le format diffusers (dossier transformer/) attendu par ai-toolkit, en
utilisant la table de mapping OFFICIELLE de ComfyUI.

Le mapping `z_image_to_diffusers` est copié verbatim depuis ComfyUI
(comfy/utils.py) — c'est la table autoritative que ComfyUI emploie pour charger
les modèles diffusers Z-Image. Aucune devinette : renommage des couches groupées
(all_x_embedder.2-1, all_final_layer.2-1) + split de l'attention QKV fusionnée
en to_q/to_k/to_v.

Usage (lancer avec le python d'ai-toolkit pour la validation diffusers) :
  python convert_comfy_zimage_to_diffusers.py <input.safetensors> <official_config.json> [--save <out_dir>]

Sans --save : mode GATE seul (construit le state-dict diffusers en RAM et valide
les clés/shapes contre ZImageTransformer2DModel sur device 'meta', sans rien
écrire — décisif et léger).
"""
import sys
import os
import json
import shutil
import re

import torch
from safetensors.torch import load_file, save_file


# ---- ComfyUI comfy/utils.py: z_image_to_diffusers (verbatim, attribution) -----
def z_image_to_diffusers(mmdit_config, output_prefix=""):
    n_layers = mmdit_config.get("n_layers", 0)
    hidden_size = mmdit_config.get("dim", 0)
    n_context_refiner = mmdit_config.get("n_refiner_layers", 2)
    n_noise_refiner = mmdit_config.get("n_refiner_layers", 2)
    key_map = {}

    def add_block_keys(prefix_from, prefix_to, has_adaln=True):
        for end in ("weight", "bias"):
            k = "{}.attention.".format(prefix_from)
            qkv = "{}.attention.qkv.{}".format(prefix_to, end)
            key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, hidden_size))
            key_map["{}to_k.{}".format(k, end)] = (qkv, (0, hidden_size, hidden_size))
            key_map["{}to_v.{}".format(k, end)] = (qkv, (0, hidden_size * 2, hidden_size))
        block_map = {
            "attention.norm_q.weight": "attention.q_norm.weight",
            "attention.norm_k.weight": "attention.k_norm.weight",
            "attention.to_out.0.weight": "attention.out.weight",
            "attention.to_out.0.bias": "attention.out.bias",
            "attention_norm1.weight": "attention_norm1.weight",
            "attention_norm2.weight": "attention_norm2.weight",
            "feed_forward.w1.weight": "feed_forward.w1.weight",
            "feed_forward.w2.weight": "feed_forward.w2.weight",
            "feed_forward.w3.weight": "feed_forward.w3.weight",
            "ffn_norm1.weight": "ffn_norm1.weight",
            "ffn_norm2.weight": "ffn_norm2.weight",
        }
        if has_adaln:
            block_map["adaLN_modulation.0.weight"] = "adaLN_modulation.0.weight"
            block_map["adaLN_modulation.0.bias"] = "adaLN_modulation.0.bias"
        for k, v in block_map.items():
            key_map["{}.{}".format(prefix_from, k)] = "{}.{}".format(prefix_to, v)

    for i in range(n_layers):
        add_block_keys("layers.{}".format(i), "{}layers.{}".format(output_prefix, i))
    for i in range(n_context_refiner):
        add_block_keys("context_refiner.{}".format(i), "{}context_refiner.{}".format(output_prefix, i))
    for i in range(n_noise_refiner):
        add_block_keys("noise_refiner.{}".format(i), "{}noise_refiner.{}".format(output_prefix, i))

    MAP_BASIC = [
        ("final_layer.linear.weight", "all_final_layer.2-1.linear.weight"),
        ("final_layer.linear.bias", "all_final_layer.2-1.linear.bias"),
        ("final_layer.adaLN_modulation.1.weight", "all_final_layer.2-1.adaLN_modulation.1.weight"),
        ("final_layer.adaLN_modulation.1.bias", "all_final_layer.2-1.adaLN_modulation.1.bias"),
        ("x_embedder.weight", "all_x_embedder.2-1.weight"),
        ("x_embedder.bias", "all_x_embedder.2-1.bias"),
        ("x_pad_token", "x_pad_token"),
        ("cap_embedder.0.weight", "cap_embedder.0.weight"),
        ("cap_embedder.1.weight", "cap_embedder.1.weight"),
        ("cap_embedder.1.bias", "cap_embedder.1.bias"),
        ("cap_pad_token", "cap_pad_token"),
        ("t_embedder.mlp.0.weight", "t_embedder.mlp.0.weight"),
        ("t_embedder.mlp.0.bias", "t_embedder.mlp.0.bias"),
        ("t_embedder.mlp.2.weight", "t_embedder.mlp.2.weight"),
        ("t_embedder.mlp.2.bias", "t_embedder.mlp.2.bias"),
    ]
    for c, diffusers in MAP_BASIC:
        key_map[diffusers] = "{}{}".format(output_prefix, c)
    return key_map


PREFIX = "model.diffusion_model."


def build_diffusers_state_dict(comfy_path):
    raw = load_file(comfy_path)
    sd = {(k[len(PREFIX):] if k.startswith(PREFIX) else k): v for k, v in raw.items()}
    n_layers = max([int(m.group(1)) for k in sd if (m := re.match(r"layers\.(\d+)\.", k))], default=-1) + 1
    n_ref = max([int(m.group(1)) for k in sd if (m := re.match(r"context_refiner\.(\d+)\.", k))], default=-1) + 1
    n_noise = max([int(m.group(1)) for k in sd if (m := re.match(r"noise_refiner\.(\d+)\.", k))], default=-1) + 1
    qkv_rows = sd["layers.0.attention.qkv.weight"].shape[0]
    # // 3 suppose MHA (qkv fusionne = 3*dim, n_kv_heads == n_heads). Vrai pour
    # Z-Image (30 heads, pas de GQA). On asserte la divisibilite : un format GQA
    # (qkv = dim + 2*kv_dim) donnerait une dim fausse + des slices q/k/v decalees.
    if qkv_rows % 3 != 0:
        raise RuntimeError(
            f"qkv.weight rows={qkv_rows} non divisible par 3 : checkpoint non-MHA "
            f"(GQA ?) non supporte par ce convertisseur Z-Image.")
    dim = qkv_rows // 3
    print(f"  derived config: n_layers={n_layers} n_context_refiner={n_ref} n_noise_refiner={n_noise} dim={dim}")
    if n_noise != n_ref:
        # z_image_to_diffusers utilise UN seul n_refiner_layers pour les DEUX refiners.
        # Si les profondeurs different, la table n'emet que min(n_noise, n_ref) couches :
        # les couches reelles en trop (ex. noise_refiner.n_ref..n_noise-1) seraient
        # SILENCIEUSEMENT abandonnees, et le GATE (qui compare a un modele cfg.json a
        # profondeur unique) afficherait quand meme PASSED. Le modele diffusers cible
        # ne peut de toute facon pas representer des profondeurs asymetriques -> on
        # refuse fort plutot que de produire un transformer incomplet en vert.
        raise RuntimeError(
            f"noise_refiner depth ({n_noise}) != context_refiner ({n_ref}) : profondeurs "
            f"asymetriques non representables en diffusers ZImageTransformer2DModel "
            f"(n_refiner_layers unique). Conversion refusee (poids seraient perdus).")
    key_map = z_image_to_diffusers({"n_layers": n_layers, "dim": dim, "n_refiner_layers": n_ref})

    out, unmapped = {}, []
    for diff_key, src in key_map.items():
        if isinstance(src, tuple):
            ck, (d, start, length) = src
            if ck in sd:
                out[diff_key] = sd[ck].narrow(d, start, length).contiguous().clone()
            else:
                unmapped.append((diff_key, ck))
        else:
            if src in sd:
                out[diff_key] = sd[src]
            else:
                unmapped.append((diff_key, src))
    used = {(s[0] if isinstance(s, tuple) else s) for s in key_map.values()}
    extra = [k for k in sd if k not in used]
    print(f"  mapped {len(out)} diffusers keys | {len(unmapped)} src-absent | {len(extra)} comfy keys unused")
    for mk in unmapped[:8]:
        print("     SRC-ABSENT:", mk)
    for e in extra[:8]:
        print("     UNUSED-COMFY:", e)
    return out


def gate(out_sd, cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    try:
        from diffusers import ZImageTransformer2DModel
    except ImportError:
        from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
    with torch.device("meta"):
        model = ZImageTransformer2DModel.from_config(cfg)
    exp = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    got = {k: tuple(v.shape) for k, v in out_sd.items()}
    missing = [k for k in exp if k not in got]
    unexpected = [k for k in got if k not in exp]
    mism = [(k, exp[k], got[k]) for k in exp if k in got and exp[k] != got[k]]
    print(f"\n=== GATE === model keys={len(exp)} | converted={len(got)}")
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}  shape_mismatch={len(mism)}")
    for m in missing[:15]:
        print("     missing:", m)
    for u in unexpected[:10]:
        print("     unexpected:", u)
    for k, e, g in mism[:10]:
        print(f"     shape: {k} expected {e} got {g}")
    # `unexpected` DOIT compter dans le verdict : diffusers/ai-toolkit chargent en
    # strict=True et levent RuntimeError sur la moindre cle surnumeraire. Un GATE qui
    # ignore `unexpected` peut afficher PASSED puis sauver un .safetensors inchargeable
    # (ex. biais d'attention / adaLN context_refiner emis par la table mais absents du
    # modele). On echoue donc aussi des qu'il y a des cles inattendues.
    ok = (len(missing) == 0 and len(mism) == 0 and len(unexpected) == 0)
    print("\n[GATE PASSED] toutes les cles diffusers remplies, shapes OK, aucune cle en trop" if ok
          else "\n[GATE FAILED] remap incomplet (manquantes / shapes / cles inattendues)")
    return ok


def main():
    inp, cfg_path = sys.argv[1], sys.argv[2]
    save_dir = None
    if "--save" in sys.argv:
        save_dir = sys.argv[sys.argv.index("--save") + 1]
    print(f"Loading {inp} ...")
    out_sd = build_diffusers_state_dict(inp)
    ok = gate(out_sd, cfg_path)
    if ok and save_dir:
        tdir = os.path.join(save_dir, "transformer")
        os.makedirs(tdir, exist_ok=True)
        save_file(out_sd, os.path.join(tdir, "diffusion_pytorch_model.safetensors"))
        shutil.copy2(cfg_path, os.path.join(tdir, "config.json"))
        print(f"\nsaved diffusers transformer -> {tdir}")


if __name__ == "__main__":
    main()
