# Knowledge base layout and document format

Knowledge content is deliberately split into separate roots:

- `approved/` is the only content searched by the application by default. Place only approved, current store/customer-support information there.
- `demo/` contains conspicuously fictional samples for local quality tests. It is indexed into a different Pinecone namespace only with the explicit `--demo` command and is never searched by normal application requests.

Supported files are UTF-8 Markdown (`.md`) and plain text (`.txt`). The recommended format begins with this metadata header:

```text
---
document_id: return-policy-en-v1
category: return_policy
language: en
last_updated: 2026-11-14
status: approved
---
# Return policy
Approved content starts here.
```

Use ISO `YYYY-MM-DD` dates. Categories can be `store_faq`, `promotion`, `shipping_policy`, `return_policy`, `refund_policy`, or `store_information`; other lowercase category names are accepted. Language uses a short BCP-47-like tag such as `en` or `hi`. `document_id` must be unique across files and stable while maintaining the same logical document. The relative filename is recorded as `source`; each chunk stores document ID, filename, category, language, source, update date, content hash, and chunk number.

If the metadata header is omitted, the loader derives a stable document ID from the relative path, uses the parent directory as category, and records language/date as `und`/`unknown`. Explicit headers are recommended for maintainability and multilingual evaluation.

Never put secrets or customer records here. Do not place demo content under `approved/`. Do not represent the `demo/` examples as retailer policies or use their namespace in production. Configure and index approved material only after its owner has approved the exact text.
