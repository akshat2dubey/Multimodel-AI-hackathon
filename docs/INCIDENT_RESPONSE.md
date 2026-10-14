# Incident response

Use this checklist if a credential or run artifact is exposed or tampered with.

1. Stop active work with the dashboard's token-protected **Cancel run** action.
2. Revoke the exposed Copernicus or ohsome key and issue a replacement. Update only the local `.secrets/` file.
3. Rotate `JALNETRA_PIPELINE_TOKEN`, the dashboard password, and the Flask secret if they may have been exposed.
4. Preserve `logs/audit.jsonl`, `logs/jalnetra.log`, and the affected run's manifest before cleanup. Verify the audit chain with `python scripts/verify_audit.py`.
5. Remove exposed credentials from local logs and caches after preserving incident evidence. If a secret entered Git history, coordinate a history rewrite and force-push with repository owners; never do this without their approval.
6. Record the timeline, affected runs, credentials rotated, and any notifications required by the data or service provider.

The audit chain is tamper-evident, not tamper-proof. An attacker who can rewrite the whole file can recompute the hashes; keep a protected copy outside the machine for stronger evidence.
