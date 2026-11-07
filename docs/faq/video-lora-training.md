# Can I train a video LoRA (Wan 2.2, LTX-2, MiniMax H3)?
<!-- description: Video LoRA training in LDS is experimental: the beta Video lane plugin cuts clips to Wan, LTX and MiniMax H3 frame rules and trains with ai-toolkit. -->

Experimentally, with the [Video lane](../../bundled/video/) plugin, which is in beta. It cuts clips to the frame counts each model accepts (4n+1 for Wan, 8n+1 for LTX, 17n+5 for MiniMax H3), measures silence and audio level, shows burned-in subtitles and letterbox bars before training, and trains with ai-toolkit locally or on a rented GPU. Nine targets are offered across Wan 2.1, Wan 2.2, LTX-2, LTX-2.3 and MiniMax H3, but only Wan 2.2 14B has been through a finished run so far, filling 24 GB with CPU offload.

## Build a video dataset

- **+ New dataset → Video**: choose the target model, the clip length and the size, then add local files in **Add videos**. Uploads accept 200 MB per file and 1 GB per upload. Files too short for the frame count are skipped, and long sources can be split into several clips.
- A **Video Bank** reads a folder of long videos in place, without writing to it, finds the cuts with TransNetV2 and lets you triage shots before anything is encoded. Only the clips you keep are re-encoded, when you promote them.
- The clip-length menu only offers counts the target can ingest. The trainer does not check the count itself: an off-rule clip silently loses its trailing frames.

| Target | Frame rule | Frame rate |
|---|---|---|
| Wan 2.1 T2V (1.3B / 14B), Wan 2.1 I2V 14B | 4n+1 | 16 fps |
| Wan 2.2 T2V A14B, Wan 2.2 I2V A14B | 4n+1 | 16 fps |
| Wan 2.2 TI2V-5B | 4n+1 | 24 fps |
| LTX-2, LTX-2.3 | 8n+1 | 24 fps |
| MiniMax H3, MiniMax H3 Ref2V | 17n+5 | 24 fps |

## Checks before training

- **Sound:** for the targets that keep an audio track (LTX and MiniMax H3), every shot is scored for its share of silence and its level in dBFS, because a set of silent clips teaches the model to be silent.
- **Subtitles and bars:** an optional pass reads three frames per shot, measures letterbox bars and text that holds still, and reports the crop that would remain. Reading text needs a small CPU package from Setup; without it the pass measures the bars only and says so.
- **Encoding damage:** one sweep per file measures duplicated frames, compression blocking and the edge blur that gives away an upscale.
- **Captions:** a pass describes what happens in each shot, as editable prose. Every promoted clip gets a `.txt` file, empty when it has no caption, and the build dialog counts the empty ones before encoding.

## Training and testing

**Train this dataset** runs the set through the ai-toolkit installed on your machine, and **Train in the cloud** rents a GPU for the same set, showing the GPU and its hourly price once the pod reports them. Measured on the one finished run, Wan 2.2 14B filled 24 GB at 170 to 185 seconds per step, with the CPU offload that makes 24 GB possible at all. The plugin's Video Test Studio generates MiniMax H3 clips from an image or from text to compare LoRAs.

## Limits

- Five targets have no verified base and need you to name a base repository: both Wan I2V variants, Wan 2.2 TI2V-5B and both LTX-2 versions. The cloud lane refuses them with the reason, and that base can only be supplied through the API today.
- Wan 2.2 A14B saves each checkpoint as a high-noise and low-noise pair; either file alone cannot be loaded.
- MiniMax H3 needs about 43 GB of weights from `Comfy-Org/MiniMax-H3`, and LDS asks before downloading them. MiniMax H3 Ref2V also needs 1 to 4 reference photos attached to the dataset.
- The MiniMax H3 Community License grants no rights in the EU, the UK, South Korea or the USA, and the restriction covers the outputs as well as the model. Check your territory before using it.
- Near-duplicate shots are grouped with a threshold carried over from images, not calibrated on video. There is no audio captioning, and scene search ranks shots rather than filtering them.

## Related

- [How much VRAM do I need to train a LoRA?](lora-training-vram.md)
- [Which LoRA trainer should I use? Do I need the command line?](which-lora-trainer.md)
- [Video Bank: capabilities and limits](../guide/features.md#video-bank-beta--first-release-read-the-limits)
