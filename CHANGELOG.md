# Changelog

## Unreleased

- Reworked the Streamlit UI styling to improve light-mode contrast, tighten visual hierarchy, and add reusable skeleton-loading states.
- Added LM Studio sidebar settings for base URL, API key, timeout, retry count, and preferred model override.
- Hardened LM Studio requests with endpoint-aware client creation, exponential backoff, streaming completion support, and balanced JSON parsing.
- Upgraded HR scoring prompts to use weighted rubric dimensions and richer structured output, including score breakdowns and stronger recommendation labels.
- Upgraded Candidate Mode prompts to emphasize ATS-friendly formatting, STAR-style bullets, quantified impact, action verbs, and categorized skills.
- Expanded Candidate Mode resume generation to support a richer JSON schema and improved PDF rendering order and typography.
- Added pasted-resume support in HR Mode alongside PDF uploads, with batch processing and empty/oversized file handling.
- Improved self-simulation and screening views to show richer score breakdowns and more explainable candidate reports.
- Kept all existing local-first behavior and preserved the existing embedding-based screening flow.
