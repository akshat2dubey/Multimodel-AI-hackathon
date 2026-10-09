# FloodMapper — Status Report to CEO

**From:** Lead Developer (codebuff / Buffy)
**To:** CEO
**Date:** 9 October 2026
**Project:** Track B — Bhote Koshi–Trishuli flood mapping pipeline
**Overall status:** 🟡 **ON TRACK, ONE BLOCKER**

---

## 1. Where we are

The engineering is ahead of the data. We have a **working end-to-end pipeline** — satellite
ingestion, terrain analysis, infrastructure overlay, damage & cut-off analysis, report
generation, and a live web dashboard — that runs on command today. 9 of our 10 dashboard
routes return HTTP 200, and they've been hit live against a running server, not assumed.

What's real:
- **A real 258 MB satellite elevation model** of the Trishuli corridor, pulled from the
  public Copernicus AWS Open Data bucket — no account needed. 6 tiles, 38–45 MB each,
  covering 84–87°E × 27–29°N. Marked `"synthetic": false` in its own sidecar file.
- **A real flood-flow trace** computed off that real DEM: 9,171 points, 251 km of channel,
  5,961 m of descent, starting around 5,431 m up and reaching 234 m. Written as a verified
  GeoJSON that the dashboard serves — 343 KB, 0.04 seconds — labelled honest about what it
  is.

What's placeholder, and we're honest about it:
- Everything from the satellite imagery onward is **synthetic demo data** fabricated so the
  code, the dashboard, and the report could be built and tested before the imagery arrives.
  The flood mask, the damage numbers, the cut-off list, the report itself — all labelled
  synthetic in their own metadata. We have a `DEMO_DATA_NOTICE.txt` calling this out plainly,
  and the report.md states the caveats. Nobody can accidentally quote a made-up number as a
  real measurement because the system keeps flagging them.
- `models/` is **empty** — no trained model exists yet.

So the honest summary is: **the engine works. The fuel isn't here yet.**

---

## 2. What's done

| Capability | Status |
|---|---|
| Terrain pipeline (DEM download + flow trace) | ✅ Done and verified on real data |
| Sink-breaching / fallback D8 so the tracer doesn't stall on flat noise | ✅ Done |
| Dashboard with 10 routes, all checked live | ✅ Done |
| Map UI with flood layer, roads, buildings, bridges, flow path | ✅ Done |
| Synthetic data generator (so the code has something to demo on) | ✅ Done |
| Report generator (renders a full situation report) | ✅ Done |
| Pipeline orchestration (one command → results) | ✅ Done |
| Documentation — README, report.md, plus the full technical tutorial | ✅ Done |

In short: every piece of the pipeline exists, runs, and talks to every other piece. The
architecture is done. We built the car before we bought the fuel, which is the right order.

---

## 3. What's left — and why

### The one thing that blocks everything downstream

**We need a Copernicus Data Space account and an OAuth client.** That's free, takes about
15 minutes, and only the account holder can do it — you can't delegate it and the secret
shouldn't go in chat.

Why it matters: the Sentinel-1 radar + Sentinel-2 optical imagery comes from Copernicus via
an OAuth client-credentials grant. Without the client ID and secret, our download scripts
can't authenticate, so **the three headline outputs — real flood mask, real damage
assessment, real cut-offs — stay synthetic.** Everything after that (training the model,
running inference, regenerating the report with real numbers) chains off that.

### Decisions only you can make

- **Confirm the case-study location is 28.3°N / 85.7°E** (the Bhote Koshi–Trishuli corridor,
  Nepal), not the 78.7°E the early README accidentally had. We caught and fixed 78.7°E in
  the docs and in the dashboard's own pipeline trigger, but I want you to sign off: if the
  real event you're presenting is somewhere else, everything downstream has to be re-pointed.
- **Choose the training data source for the model.** The repo can pull from three options —
  Hugging Face GeoTIFFs (recommended — simplest mapping to what the training code expects),
  Hugging Face WebDataset, or a Dropbox SLC bundle. Tell me "GeoTIFFs" and I'll build the
  converter.
- **OK the GB-scale Kuro Siwo download.** It's large; I won't pull it without your go.

### Deliveries only you can ship

- The **3-minute demo video** — narration, face, screen recording. Not something I can do.
- The **hackathon portal submission** itself — team registration, upload, hitting the
  deadline. That's on you.

### Housekeeping I'll take care of when you say the word

- Clean up the five stray debug logs in `dashboard/` and one superseded backup GeoJSON.
  They don't hurt anything, but they're clutter and I'd rather not ship them.
- Push the repo publicly (nothing has been pushed yet; I won't do it until you confirm).

---

## 4. Honesty note — what not to present as fact

A few numbers currently floating around in the synthetic results that **must never be quoted
as measurements of the actual Nepal event**:

> 6,740.79 ha total water · 5,278.36 ha flood extent · 78.3% flood coverage
> · 68 cut-off settlements · 70.8% cut-off rate · 4 destroyed buildings · 3 damaged bridges

Those are properties of the demo fixture, not the earthquake event. They exist so the UI has
something to render, and the system keeps flagging every one of them as `"synthetic": true`.
Until real Sentinel imagery downloads and the model runs against it, presenting any of those
as findings would be wrong, and I've written that into the docs three different ways so it
doesn't slip through on stage.

The one number I *would* put in front of a judge: the **251 km flood path from the real DEM**.
It's real, verified, and I'd show it as "the least-obstruction drainage route from the satellite
terrain" — not a surveyed channel. 1,201 short sink-breaches were needed because strict
D8 stalls on flat pixels, and I've documented that honestly. It's the right call for this
terrain, and I'd rather present a real-but-caveated trace than a fake-but-clean number.

---

## 5. Risk that keeps me up a little — security

The dashboard was built as a laptop demo. As shipped, it is **not safe to put on venue Wi-Fi
or touch a public interface**, and I want to be up front about why.

Three things are true as it stands:

1. **It listens on all network interfaces.** The server log says `Running on all addresses
   (0.0.0.0)` and broadcast itself on our LAN IP. Anyone on the same Wi-Fi can reach it.

2. **The Flask debugger is on by default.** The Werkzeug debugger console is a live Python
   REPL anyone who hits a traceback can use. With the dashboard open on your network, that's
   remote code execution — someone else gets a Python console on your machine. This is not a
   theoretical concern; it's exactly the vulnerability people have pwned dev servers with for
   years.

3. **There is no access control on the APIs.** Any website you visit while the dashboard is
   running can read every result — I tested this live: a request with a fake `Origin` header
   came back with that origin reflected in the `Access-Control-Allow-Origin` header. So a
   hostile page could read our flood extents, damage numbers, everything, cross-site.

None of this matters while the demo stays on localhost on your own machine. It matters the
moment we demo on shared Wi-Fi, or worse, deploy. I have three lines of fix for the first
three — bind to `127.0.0.1`, default `debug=False`, restrict CORS to `http://localhost:5000`
— and a full hardening backlog for if we ever go public. I'd rather do the small fix now,
before anyone sees the demo, than explain a pwned laptop after.

There's also a housekeeping risk: **there's no `.gitignore` and no git repository at all yet,
so it's a matter of time before a 258 MB DEM or a secret file gets committed by accident.**
That's an easy fix and I'll make it as part of hardening, but I'd rather flag it now than let
it bite us.

---

## 6. What I need from you

**One decision, and one account:**

1. **Create the Copernicus account + OAuth client (15 minutes).** I can walk you through it
   step by step — register at `dataspace.copernicus.eu`, create an OAuth client in the Sentinel
   Hub dashboard, copy the client ID and secret (shown only once), save them to a local
   `.secrets/cdse.env` file that never touches chat or a command line, and then I verify them
   live against the official auth endpoint. That single thing unblocks the whole remote-sensing
   half.

2. **Give me the go on the Kuro Siwo source — "GeoTIFFs" is my recommendation** — and OK the
   GB-scale download. I'll build the converter and pull it.

**Two line items if you want, otherwise I'll hold:**

- Sign off that 28.3°N / 85.7°E is the right case-study location (vs. any other coordinate you
  have in mind).
- Say whether you want me to do the three-line security fix now before any demo, or hold it.

---

## 7. My read on timing and confidence

If you create the Copernicus account this week, I can run the real Sentinel-1 and Sentinel-2
downloads, confirm they land, and regenerate the real flood mask, damage, and cut-off numbers —
which puts us in genuinely shippable shape.

Everything before that — the terrain pipeline, the dashboard, the report engine, the docs, the
synthetic demo data so the whole thing can be exercised today — is done and verified. We're not
at risk on the build side. We're at risk only on the one external dependency, and it's something
you control in 15 minutes.

Confidence on the build: high. Confidence on the data: zero until Copernicus is in. The two are
independent and the build's the harder half, in my view — once the imagery lands, the pipeline
is there and waiting.

---

*— Buffy, Lead Developer*
