export const CLOUD_WHATS_NEW = [
  {
    id: '2026-09-26-cloud-video-rank',
    date: '2026-09-26',
    title: 'Keep your chosen rank in cloud video training',
    blurb: 'Cloud video launches accept the rank selected in LDS, validate it before reserving a GPU, and retain it in the training configuration, retries and continuations.',
  },
  {"id": "2026-09-23-cloud-video-sample-choice", "date": "2026-09-23", "title": "Keep all your video training sample prompts", "blurb": "Cloud video training preserves every requested sample prompt instead of refusing more than four. Each prompt adds a video generation at every sampling interval, so larger selections take longer on the rented GPU."},
  {
    id: '2026-09-23-cloud-test-family',
    date: '2026-09-23',
    title: 'Test cloud checkpoints in their trained image family',
    blurb: 'Test in Studio now opens the image family used by the selected cloud run, including FLUX.1, Anima and Qwen-Image 2.1. With the matching LDS update, datasets trained for several families open the intended checkpoints directly.',
  },
  {
    id: '2026-09-23-cloud-rental-reliability',
    date: '2026-09-23',
    title: 'Avoid incompatible GPU hosts and release failed rentals sooner',
    blurb: 'GPU offers now match the training image’s CUDA requirement. Known failed hosts remain excluded, and confirmed container or CUDA startup errors release a broken rental promptly instead of waiting for a long silence timeout. Slow downloads that keep progressing are preserved.',
  },
  {
    id: '2026-09-23-cloud-qwen-image-21',
    date: '2026-09-23',
    title: 'Train Qwen-Image 2.1 on a cloud GPU',
    blurb: 'Image-and-caption datasets can now train Qwen-Image 2.1 LoRAs on a dedicated AI Toolkit pod image. The GPU picker enforces this model’s memory, disk and GPU requirements and shows live hourly prices.',
  },
  {
    id: '2026-09-22-cloud-continue-gpu-choice',
    date: '2026-09-22',
    title: 'Choose your GPU when continuing training in the cloud',
    blurb: 'Continue training now lists available GPU models, VRAM and live hourly prices before starting. Pick a card when moving a local checkpoint to the cloud or continuing a previous cloud run, from Checkpoints or Runs.',
  },
{
    id: '2026-09-05-zz-vast-referral-link',
    date: '2026-09-05',
    title: 'Creating a vast.ai account through the app now supports the project, at no cost to you',
    blurb: 'Every vast.ai link in the app and its docs is a referral link: vast.ai pays this project 3% of what a referred account spends there. Prices are identical, nothing in the app changes, and the disclosure sits beside the account-creation steps.',
    to: '/settings/training',
  },
{
    id: '2026-09-02-video-cloud-launch-window',
    date: '2026-09-02',
    title: 'See the price, the time and the total before renting a GPU for a video run',
    blurb:
      'A video set used to rent a pod on one click, with the cost hidden inside a '
      + 'closed dropdown. ☁ Train in the cloud now opens the same window as an image '
      + 'dataset: every GPU class with its price per hour, a rough duration and total '
      + 'for this set, a warning when the run would outlive the runtime cap, and this '
      + 'month’s spend against your budget. A readiness card above the launch says what '
      + 'still stands in the way — no clips, no references, a missing key — before '
      + 'anything is spent, and “second pod, billed separately?” is a question you can '
      + 'answer instead of an error.',
    to: '/datasets',
  },
{
    id: '2026-08-30-video-cloud-gpu-picker',
    date: '2026-08-30',
    title: 'Pick the GPU for a cloud video run — with prices in front of you',
    blurb:
      'Cloud video training used to rent the cheapest suitable card without '
      + 'showing you a number. The panel now lists one offer per GPU class — '
      + 'price per hour, VRAM, and a rough time and cost for your dataset — '
      + 'and lets you pick, or leave it on “cheapest suitable” as before. '
      + 'Estimates are honest about being rough: they come from one measured '
      + 'run, and a clip length nothing was measured at shows no estimate '
      + 'rather than an invented one.',
    to: '/datasets',
  },
{
    id: '2026-08-29-video-lora-in-the-cloud',
    date: '2026-08-29',
    title: 'Train a video LoRA on a rented GPU — MiniMax H3 included',
    blurb:
      'Promote clips out of a 🎬 video bank, pick a target model, and '
      + '☁️ Train in cloud now rents a pod that can actually run it: the '
      + 'right ai-toolkit, enough VRAM, and enough disk for a base that weighs '
      + '42 GB — all settled before the rental instead of discovered after it. '
      + 'MiniMax H3 has now been trained end to end that way, on clips cut to '
      + 'its own geometry (107 frames at 24 fps, audio kept at 32 kHz stereo), '
      + 'and the checkpoints come back to the dataset. Check your territory '
      + 'first: H3’s licence grants no rights in the EU, the UK, South Korea '
      + 'or the USA without MiniMax’s free authorisation, and it covers what '
      + 'you generate as well as the model.',
    to: '/datasets',
  },
{
    id: '2026-08-17-vast-do-not-rent-this-machine-again',
    date: '2026-08-17',
    title: 'Stopping a cloud run can now blacklist the machine',
    blurb:
      'The app already refuses to re-rent a vast host when it can see the failure — a boot that never completes, a pod that stops progressing, a checkpoint it cannot serve. What it cannot see is a machine that boots perfectly and then simply trains at half speed: no failure happens, so nothing gets banned and the picker can rent it straight back. Stopping a run from the Runs page now offers "Do not rent this machine again", which puts that host on the same blacklist for the same few days. It is a tick box, not a rule: a stop can just as easily mean you changed your mind, and that says nothing about the machine. (Asked for by mr.arrow on Discord.)',
    to: '/cloud',
  },
{
    id: '2026-08-07-parallel-cloud-runs',
    date: '2026-08-07',
    title: 'Train the same dataset twice at once',
    blurb: 'Launch a second cloud run on a dataset that is already training to compare toolkit settings side by side — confirm the extra pod, then follow each run from its own chip on the Training panel. The runs warn you if the dataset changed between launches.',
  },
{
    id: '2026-08-04-full-model-run-no-longer-reads-as-gone',
    date: '2026-08-04',
    title: 'A full-model run whose model is on Hugging Face no longer shows up as “gone” — and the app no longer offers to delete it',
    blurb:
      'The canvas asked one question to decide whether a run still had anything: is there a checkpoint file on this disk? That question has no good answer for a full model delivered to a private Hugging Face repository — there is no local file, and there never was. So those runs were drawn dimmed, badged “gone”, and given a “Remove this run” button under the words “No checkpoints left on disk”, for a model that was perfectly fine and had cost hours of GPU. Removing one threw away the lineage, the notes and the only record of which repository the model was in. A full-model run is now asked about both of the addresses it can have: it shows “💾 full model here” when the weights are on this computer, “☁ on Hugging Face” when they are in its repository, and it is only offered for removal when the model is genuinely gone from both. If you try anyway, the app now says where the model still is instead of deleting the trail to it.',
  },
{
    id: '2026-08-04-full-model-lands-on-your-computer',
    date: '2026-08-04',
    title: 'A finished full model now lands on YOUR computer — and a full Hugging Face quota can no longer end a training',
    blurb:
      '🖥 Until now a full-model (dense) run had exactly one address: a private Hugging Face repository the pod pushed to while it trained. That address has a ceiling nobody controls, and it collected: a run died 250 steps from the end on “403 private repository storage limit reached”, after eight hours of paid GPU, and only survived because 50 GB were deleted by hand. So the order is reversed. The finished model is downloaded to your checkpoint folder FIRST, the ~10 GB fp8 file for ComfyUI with it, and the pod is destroyed only once the file here is proven — the byte count has to match what the pod advertised, and the safetensors header has to re-read. Only then is the master uploaded to Hugging Face as a backup, and that upload is now allowed to fail: it costs the ability to continue that model later, nothing else. Nothing is pushed while the run trains, so the quota can no longer reach the training at all. The transfer is tens of minutes of 26 GB, so it shows its progress, survives an app restart, and can be stopped and resumed without losing what already landed — and if it fails, the machine is kept and the Runs page offers “Fetch to this computer”. A launch also checks this machine’s disk before renting anything, and refuses (confirmably, like every other estimate) when the drive plainly has no room. Choose the delivery in Settings ▸ Storage ▸ Full-model delivery; runs made before today keep their Hugging-Face-only behaviour exactly as it was.',
  },
{
    id: '2026-08-04-cloud-quantize-rents-a-machine-that-fits',
    date: '2026-08-04',
    title: 'Cloud quantization picks a machine that can hold the model (server side — this lane has no button)',
    blurb:
      'A correction first: this lane has no interface, so there is no click to make. The fix below is real and lives in the server half; the reachable way to shrink a model is ✨ Quantize to fp8, locally, from a full-model card or Settings ▸ Storage. Cloud quantization used to give up one second after being asked with “create_instance failed: HTTP 400 {}” — no machine, no money spent, and no reason. Two things were wrong. It rented the cheapest offer on the market, and cheap is exactly where free disk runs out: a 26 GB model needs about 86 GB on the pod for the master, its fp8 twin and the download cache, while the top offer of a live search had 57 GB — an ask vast refuses outright. And a single refusal ended the job, even though the next machine would have taken it. Now the search only considers machines with the disk this job will claim, the offer is chosen by the same rule a training launch uses (bad hosts skipped, suspiciously cheap listings ignored), and a refusal moves to the next candidate instead of ending everything. When a rental really is impossible, the error quotes what vast said rather than an empty “{}”. The estimate stays an estimate — but if the market moved and the only machine left costs materially more than the price you agreed to, it tells you and rents nothing.',
  },
{
    id: '2026-08-03-quantize-in-the-cloud-without-downloading',
    date: '2026-08-03',
    title: 'Cloud quantization: built on the server, never given a button — corrected',
    blurb:
      'This entry announced a “☁ Quantize to fp8 in the cloud” button, and that button was never wired: the service, its three endpoints and their tests are in the app, but nothing in the interface calls them, so there has never been anything to click. We are correcting the claim rather than quietly leaving it: an app that announces what it cannot do is worse than one that shipped less. What DOES work, and always did, is the local conversion — ✨ Quantize to fp8, from a full-model recipe card or from Settings ▸ Storage. It runs on your CPU in about a minute, so the arithmetic was never the expensive part. The cost the cloud lane was meant to save is bandwidth: a master that lives only in your Hugging Face repo has to come down (26 GB) and go back up (10 GB) for the local path to touch it. If that round trip is your problem, say so on the Discord — the server half is already written and only the door is missing.',
  },
{
    id: '2026-08-03-hf-storage-precheck-and-cleanup',
    date: '2026-08-03',
    title: 'Full-model cloud runs now check your Hugging Face space before renting a GPU',
    blurb:
      'A full-model (dense) Krea run delivers each ~26 GB checkpoint straight into a private Hugging Face repo — and that push happens at the very end. A run died at step 2750 of 3000 on “private repository storage limit reached”, hours of paid GPU gone, because the account\'s private space was full of custom-base caches nothing in the app ever showed you. Now the launch measures your private storage first and refuses before a pod is rented, saying how much is missing and what is taking the room — with Train anyway always available, because Hugging Face publishes no quota endpoint and the ceiling is an estimate. Settings ▸ Training gained a Hugging Face storage card that lists every lds-base-* cache with its size, the local file it mirrors and the run that last used it, and deletes them one by one or all at once — warning you when a cache is the last copy left. And if a run hits the wall anyway, it now says so in plain words and keeps the pod so the checkpoint is recoverable.',
  },
{
    id: '2026-08-03-restart-no-longer-kills-a-live-cloud-run',
    date: '2026-08-03',
    title: 'Restarting the app no longer kills a cloud run that is training fine',
    blurb:
      'When the app restarted, it picked the run back up and asked vast.ai whether the pod still existed. If that one answer came back without the pod in it — which happens, and means nothing — the run was declared dead about ten seconds later, and the "stop" that followed reached the pod that was still training and ended it. A run at step 825 of 3000 was lost that way, with the hour already paid. Now the pod itself is asked: a pod that answers is a pod that exists, whatever the marketplace says, and silence has to last minutes before the run is given up. If it truly cannot be reached, no stop is sent to a machine we could not talk to, and the pod is kept so the result stays recoverable.',
  },
{
    id: '2026-08-02-cloud-launch-is-observable',
    date: '2026-08-02',
    title: 'A cloud launch now tells you what it is doing, and for how long',
    blurb:
      'Renting a GPU takes minutes, and the button used to say "Launching…" for all of them — impossible to tell a normal wait from a dead one. The launch now shows its steps as it walks them (preparing the dataset, searching for an offer, renting and booting the pod, uploading, starting the job) with the time elapsed, on the dataset panel and on the Runs page. While a pod boots you also see how long it is allowed to take, so a machine that never starts ends with a plain explanation instead of a frozen screen — it is released, it stops billing, and launching again picks a different host. A launch can be cancelled from the Runs page like any run.',
  },
{
    id: '2026-08-02-cloud-only-installs-see-their-checkpoints',
    date: '2026-08-02',
    title: 'Trained in the cloud without ai-toolkit? Your checkpoints show up now',
    blurb:
      'The checkpoint list refused to answer at all unless local training was set up, so an install that only ever trains in the cloud saw an empty panel — its own paid-for cloud saves were there on disk, just never displayed. The list now always answers: cloud saves appear whether or not ai-toolkit is configured, and the local half simply stays empty when there is no local trainer. Deleting a cloud-trained LoRA you had deployed to ComfyUI works from the same install too, instead of being listed but undeletable.',
  },
{
    id: '2026-08-01-verified-cloud-token-stops-nagging',
    date: '2026-08-01',
    title: 'Full-model training confirms your Hugging Face token instead of re-asking for it',
    blurb:
      'The 80 GB GPU picker used to show the same “configure a token before renting the GPU” notice even when your saved token had just been verified. It now reports the verified delivery namespace, keeps a distinct amber note for a global write token, and shows the setup instructions only when something is genuinely missing.',
  },
{
    id: '2026-07-28-cloud-boot-waits-for-a-pod-that-is-still-working',
    date: '2026-07-28',
    title: 'Cloud launches survive a slow host pulling its image',
    blurb:
      'A pod that took more than 25 minutes to boot was terminated even when it was honestly downloading its multi-gigabyte image — and its host was quietly skipped for the next three days. The boot wait now restarts its clock whenever the pod shows real progress, keeps an absolute ceiling so a dead pod still dies fast, tells you where the boot actually got to, and only exiles a slow host for a few hours.',
  },
{
    id: '2026-07-28-cloud-watchdog-counts-a-downloading-pod-as-progress',
    date: '2026-07-28',
    title: 'A cloud run is no longer killed while its pod is downloading normally',
    blurb:
      'The run card now shows the bytes a pod is fetching — but the watchdog guarding that phase was still only watching the training step counter, so a run on a slow host was killed at 45 minutes for "no progress" while the card beside it showed the download working perfectly (a 26.3 GB model at the 2.6 MB/s some hosts give you takes nearly 3 hours). The watchdog now reads the same counter the card does: bytes moving is progress, and the clock restarts. A pod that reports no bytes at all still dies as fast as before, a hard ceiling still stops a host that will never finish, and the failure message finally says what was measured instead of guessing. The idle budget and that ceiling are now in Settings → Training. Thanks to j_o_e_l. (Discord) for the report.',
  },
{
    id: '2026-07-28-cloud-run-download-bytes-and-durable-freeze-clock',
    date: '2026-07-28',
    title: 'A downloading cloud run now shows the bytes, not a frozen sentence',
    blurb:
      'While a pod fetches its base weights — 26 GB for Krea — the run card used to show one motionless line, "fetching transformer weights", for as long as it took. Nothing told a healthy download from a dead pod short of opening the vast.ai console, and people waited hours to find out. The card now reads the pod\'s own counter: how much has landed, of how much, at what speed, with the ETA. And the "no progress" warning is finally reliable — it is measured on what the pod does, so restarting the app no longer resets it, and a monitor repeating the same sentence no longer hides a frozen run.',
  },
{
    id: '2026-07-28-busy-database-no-longer-strands-a-paid-cloud-run',
    date: '2026-07-28',
    title: 'A busy database no longer strands a paid cloud run — for real this time',
    blurb:
      'Cloud runs are watched by a monitor that writes progress to the database every few seconds. When something else was writing heavily at that moment (a caption batch, a Bank import), that write could lose the lock — and the retry meant to absorb it crashed on its own error message instead, killing the monitor. The run then sat at "TRAINING" with no error, no progress and a rented GPU still billing, until the freeze watchdog or an app restart caught it up to 45 minutes later. The retry now works, and a run whose monitor does die is closed properly with its pod terminated instead of being left open.',
  },
{
    id: '2026-07-26-cloud-run-survives-a-busy-database',
    date: '2026-07-26',
    title: '💾 A busy database no longer abandons a cloud run you are paying for',
    blurb:
      'A cloud run records its progress in the local database as it goes. When something else was writing heavily at the same time — a captioning batch, a large import — that write could be refused, and the run died on the spot, three minutes in, while the rented GPU kept billing until someone noticed. Those writes now wait their turn and retry instead of killing the run.',
  },
{
    id: '2026-07-26-cloud-run-survives-a-restart-after-submit',
    date: '2026-07-26',
    title: '☁️ Restarting the app no longer destroys a cloud run that was training',
    blurb:
      'A cloud run submits its job to the pod, then records the job id. If the app restarted in the sliver of time between those two steps, the run came back not knowing it had already submitted anything — so it submitted again, the pod refused the duplicate name, and the run died as FAILED with the GPU hour already paid for. The id is now written the instant the pod accepts the job, and if a duplicate is ever refused anyway, the run reattaches to the job already on the pod and keeps polling it instead of failing. A job that was created but never actually launched is recognised as such and started for real, rather than being read as "stopped" and buried. When nothing can be salvaged, the error now tells you what to do next and says plainly that the pod is being terminated so it stops costing money.',
  },
{
    id: '2026-07-26-cloud-checkpoint-rescue-is-never-cut-short',
    date: '2026-07-26',
    title: '💾 A cloud checkpoint being brought home can no longer be lost on the way',
    blurb:
      'The safety net that shuts down a silent cloud run had one blind spot, and it was the worst one possible: the very end, when the training has succeeded and the app is pulling the finished LoRA off the pod. Some hosts serve that file in fits and starts — a big checkpoint can take a long while — and for all that time the run reported nothing, so it looked exactly like a run that had died. The pod could be terminated with the result still on it: the work done, the money spent, and nothing to show for it. The transfer now reports itself. The run card says "Downloading" and shows the megabytes climbing, so you can see it is working rather than guess, and no watchdog treats a live transfer as silence — including after you press Stop, where rescuing the checkpoint is the whole point. A transfer that genuinely dies is still caught, just as before.',
  },
{
    id: '2026-07-26-cloud-stop-that-cannot-lie',
    date: '2026-07-26',
    title: '🛑 Stop really stops the pod — and a frozen cloud run stops billing you',
    blurb:
      'A rented GPU bills by the hour whether or not anything is happening, so two things had to become impossible. First: Stop can no longer answer "ok" without doing anything. If nothing is left in a state to wind the run down — the app was restarted, the connection to the pod wedged — the pod is now terminated on the spot, and if even that fails you get an error naming the instance to destroy in the vast.ai console instead of a reassuring message. Second: a run that goes completely silent is caught from outside itself. The run card warns as soon as a training run stops reporting, and after 45 minutes of total silence the pod is shut down automatically — checkpoints already downloaded are kept. The runtime cap is enforced from that same place, so it holds even if the run\'s own supervision died. Phases that are quiet by design — booting, uploading, downloading the result — are never cut. You can change the delay, or set it to warn only, under Settings ▸ Training ▸ Cloud training.',
  },
{
    id: '2026-07-21-cloud-unreachable-grace',
    date: '2026-07-21',
    title: '☁️ Fewer cloud runs lost to a passing network blip',
    blurb:
      "A rented pod that briefly drops off the network (a vast.ai proxy hiccup mid-training) is no longer given up so quickly: the grace before a run is declared \"pod unreachable\" is now measured as real consecutive silence, not polluted by slow log/checkpoint mirroring — and it defaults to a more forgiving 6 minutes. Too twitchy or too patient for your hosts? Tune it under Settings ▸ Training ▸ Unreachable grace. Also: a transient rental refusal at pod creation now retries on a fresh offer instead of failing the launch outright.",
  },
{
    id: '2026-07-17-slider-lora-cloud',
    date: '2026-07-17',
    title: 'Train slider LoRAs in the cloud',
    blurb:
      'Concept-slider training is unlocked on the cloud GPU path, so you can build strength sliders (age, expression, style intensity…) without tying up your local card.',
  },
];
