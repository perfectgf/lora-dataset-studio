# Remove Background

Keep one subject of a picture and drop the rest, on your own computer.

Say what stays: the main subject, a rough selection, a few words or a click. SAM 3.1 finds it, a matting
model cuts its edge, and the result is a mask that can still be painted on, or a new picture. It opens from
the navigation on an imported picture, and from the dataset, Bank and Gallery viewers. The original is
copied and never changed.

## Models

Every graph runs on nodes that ship with ComfyUI. Only weights are installed, and files that are already
there are reused.

| Model | Job | Licence |
|---|---|---|
| [BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet) | Edge of photos | MIT |
| [Lucida](https://github.com/egeorcun/lucida) | Edge of graphics, logos, text and glass | MIT; its authors say some of its training data is research-only |
| [SAM 3.1](https://huggingface.co/Comfy-Org/sam3.1) | Finds what is described or clicked | SAM License |
| [RMBG 3.0](https://huggingface.co/briaai/RMBG-3.0-Comfy) (optional) | Finest edges: hair and fur | CC BY-NC 4.0, non-commercial unless licensed from Bria |

## How a cut-out is made

1. With words or clicks, SAM 3.1 finds the object and answers with a hard shape.
2. The matting model is shown the object from close up, or the whole picture when nothing was pointed at.
   A main subject that leaves room around it is cut a second time from closer.
3. SAM decides the shape and the matte its edge. Where they part by more than strands of hair, SAM's edge
   is kept.

Each step is a job of LDS's own GPU queue. Pictures, masks and brush layers stay in the plugin's data
folder; removing one sends it to the LDS trash.
