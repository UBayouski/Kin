# Nebius / NVIDIA feedback notes

Running log of friction and wins; becomes the feedback section of the submission.

| Date | Tool | Note |
|---|---|---|
| 2026-09-18 | Token Factory | Win: OpenAI client worked first try with only a base URL change; key to first response in minutes. |
| 2026-09-18 | Token Factory | Gap: no NVIDIA vision-language model is served. All four Nemotron models reject images ("This model does not support image input"), so a Physical AI project has to pair Nemotron with a non-NVIDIA VLM (Gemma 3, MiniCPM-V). A Nemotron VL endpoint would let the whole stack be NVIDIA. |
| 2026-09-18 | Token Factory | Mixed: `/v1/models?verbose=true` exposes modality and `supported_features`, which is great, but it is easy to miss (the plain list has none of it) and not fully accurate: `google/gemma-3-27b-it` is flagged `text->text` yet accepts images. |
| 2026-09-18 | Token Factory | Gap: the Nemotron landing page advertises "Nemotron 3 Nano Omni" on Token Factory, but no such model is in `/v1/models` and guessed IDs return 404. Nemotron Nano 2 VL (announced for AI Studio in 2025) is also gone. Listing exact model IDs on the marketing page would save time. |
| 2026-09-18 | Nemotron | Win: Lightning, Super, Ultra and Nano all produced a correct first tool call. Lightning at 0.3 s is fast enough for spoken conversation on a robot. Nano spent 219 reasoning tokens (3.2 s) on the same turn. |
