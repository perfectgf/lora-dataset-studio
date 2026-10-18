## LoRA Dataset Studio V2

**V2 is the maintained LDS release.** Existing ZIP installations can upgrade from **Settings → Maintenance → Update & restart**, keeping datasets, media and history. Git installations still on the former `main` branch must first [switch to `v2`](https://github.com/perfectgf/lora-dataset-studio/blob/v2026.09.23/docs/guide/migrate-to-v2.md). Optional features are plugins: open **Plugins → Store** after updating, install the ones you use and review their settings before the first run.

### Install the core, then choose your plugins

Download **`LoRA-Dataset-Studio-windows.zip`** below, extract it into a new folder and double-click **`start.bat`**. The launcher prepares a compatible Python environment. Complete the core Setup, then open **Plugins → Store** to install the features you want. Each plugin has its own settings and preparation steps, including required ComfyUI custom nodes. Importing and organising datasets does not require a plugin.

**Free public plugins are available through the Store.** Each plugin describes its features and preparation requirements. Paid API engines and rented GPUs still charge separately when you choose to use them.

The core and these public plugins are available at no charge, with their source published under the project's PolyForm Noncommercial license. New LDS versions reach [Patreon](https://www.patreon.com/c/Loraperfectgf) members (**LDS Member**) first, then everyone, and some plugins are members-only; the version you have stays free and keeps working. Usage statistics are optional and off by default; the app asks before sharing, and you can change your choice in **Settings → Maintenance**.

### Two ways to install and update

**ZIP installations:** **Update & restart** installs the next published release, preserving your data and local configuration. The ZIP below contains the core; plugins are installed and updated separately through the Store.

**Git installations:** `v2` is the default and maintained branch. **Update & restart** follows the commits of your checkout's configured branch. The former `main` branch is now the frozen `v1` branch; it receives no further updates. Follow the [migration instructions](https://github.com/perfectgf/lora-dataset-studio/blob/v2026.09.23/docs/guide/migrate-to-v2.md) if your checkout still tracks `main`.

```text
git clone https://github.com/perfectgf/lora-dataset-studio.git
cd lora-dataset-studio
start.bat
```
