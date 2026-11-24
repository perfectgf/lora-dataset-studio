const CHAPTER = 'remove_background.guide'

export const GUIDE = {
  chapters: [{ id: CHAPTER, title: 'Remove Background',
    description: 'Keep one subject of a picture and drop the rest, on your own computer.' }],
  sections: [
    { chapter: CHAPTER, anchor: 'remove-the-background', markdown: `## Remove the background

Open **Remove background** from the navigation and import a picture, or press **Remove background** in a dataset, Bank or Gallery viewer. The original is copied and never changed.

**What stays** has four ways to say it:

- **Main subject**: the models decide. A subject that leaves room around it is cut a second time from closer.
- **Inside selection**: draw roughly around what stays. Hair may pass a little beyond your line.
- **Describe it**: a few words, such as *woman in red* or *dog, ball*. SAM 3.1 finds everything that matches.
- **Click on it**: click on what stays. Switch to **Leave out**, or hold Alt, to click on what goes. Press a dot again to take it away.

**Kind of picture** chooses the model that cuts the edge: **Photo** (BiRefNet) is quick, **Photo, finest edges** (RMBG 3.0) keeps single strands of hair and fur, **Graphic, logo or glass** (Lucida) keeps what is see-through.

With Describe and Click, SAM 3.1 decides the shape and the matting model its edge. Where the two disagree by more than strands of hair, SAM's edge is kept.

**Touch up** shows the picture with what is removed in red. Paint with **Keep** or **Remove**; Alt swaps them. **Undo** takes back the last stroke and **Reset** goes back to the models' mask. Use the wheel, or two fingers, to zoom.` },
    { chapter: CHAPTER, anchor: 'the-result', markdown: `## The result

Choose a transparent, white, grey or black background. **Clean edge colours** takes the old background's colour out of hair and soft edges. **Download** saves the cut-out at the picture's own size and **Download mask** saves the mask in black and white.

**Add to this dataset** files the cut-out as a candidate to keep or reject, next to the picture it came from. From the Bank, the Gallery or an imported picture, **Add to a dataset** asks which one. A training picture cannot be transparent, so these need a plain background.` },
    { chapter: CHAPTER, anchor: 'models', markdown: `## Models

Everything runs in your own ComfyUI, on nodes that ship with it; nothing is uploaded and no custom node is installed. A recent ComfyUI is needed. The missing files are installed from the editor or from **Plugins → Remove Background → Settings**, and files that are already there are reused.

| Model | Job | Size | Licence |
|---|---|---|---|
| BiRefNet | Cuts the edge of photos | 444 MB | MIT |
| Lucida | Cuts the edge of graphics, logos, text and glass | 885 MB | MIT |
| SAM 3.1 | Finds what you describe or click | 1.75 GB | SAM License |
| RMBG 3.0 (optional) | Cuts the finest edges | 8.41 GB, plus a 4.86 GB text encoder and a 336 MB VAE if missing | CC BY-NC 4.0: non-commercial unless licensed from Bria |` },
  ],
}
