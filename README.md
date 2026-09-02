# TrackBus AI

**AI-powered transport intelligence for public transportation**

TrackBus is an MVP-stage transport-intelligence platform designed to help public-transport operators understand **passenger flow, vehicle occupancy, crowding, demand and operational imbalance**.

TrackBus is not limited to one camera system. It is designed to work with infrastructure an operator already has — such as **APC sensors, existing bus CCTV, GPS and historical operational data** — and add TrackBus Vision only where it is useful.

> **Current status:** MVP / research prototype. The integrated dashboard, backend/API foundation, forecasting baseline, passenger beta and computer-vision pipeline are now available directly on `main`. TrackBus has **not yet been validated as a production passenger-counting system on representative Tashkent bus footage**, and no 90%+ production accuracy claim is made.

**Languages:** [English](#english) · [O‘zbekcha](#ozbekcha)

---

# English

## What problem does TrackBus solve?

Passenger demand and available fleet capacity do not always match.

One bus can be overcrowded while the next vehicle on the same route is mostly empty. Other trips may repeatedly operate far below useful capacity. Operators may already collect valuable information through APC, CCTV, GPS or payment systems, but those signals can remain fragmented across separate systems.

TrackBus aims to turn those signals into a clearer operational picture:

- How full is each bus?
- Where do passengers board and leave?
- Where and when does overcrowding repeatedly occur?
- Which route segments are repeatedly under-utilized?
- Are buses badly spaced, with one overloaded and the next mostly empty?
- How crowded may an approaching bus be when it reaches a passenger's stop?
- Which changes should be investigated: timetable, headway, dispatching, vehicle size or route allocation?

The goal is **not simply to count people**. The goal is to turn passenger-flow data into useful transport decisions.

## Platform architecture

```text
Existing APC / CCTV / TrackBus Vision / GPS / historical data
                           ↓
                  normalized events
                           ↓
                     TrackBus API
                           ↓
             occupancy + quality checks
                           ↓
            analytics + forecast baseline
                           ↓
       operator dashboard + passenger view
```

Historical payment data can later be used as an additional demand signal when access is available. Real-time payment integration is a future option and is not required for the current MVP.

## What works now?

`main` now contains the integrated TrackBus foundation, including:

- operator / Live Pilot dashboard;
- current vehicle occupancy and source-health views;
- canonical passenger-count JSON contracts;
- durable idempotent API ingestion and persistence;
- TrackBus Vision based on Ultralytics YOLO + ByteTrack;
- configurable camera calibration and crossing geometry;
- webcam, local-video and demo showcase modes;
- offline event queuing when network connectivity is unavailable;
- transparent baseline occupancy forecasting;
- passenger-facing crowding / forecast beta UI;
- detector, tracking, counting, quality, API and web tests;
- architecture, privacy, pilot, deployment and evaluation documentation.

## TrackBus Vision

TrackBus Vision performs **person detection and temporary anonymous tracking**. It does not perform facial recognition.

```text
Camera frame
   ↓
YOLO person detection
   ↓
ByteTrack temporary track IDs
   ↓
trajectory + calibrated doorway geometry
   ↓
confirmed directional crossing
   ↓
IN / OUT event
   ↓
JSON → API / database / dashboard
```

Temporary IDs exist only to follow motion during the local processing session. They are not passenger identities.

Run the presentation pipeline with:

```bash
python showcase.py --camera 0
python showcase.py --video path/to/video.mp4
python showcase.py --demo
python showcase.py --calibrate --camera 0
```

The current showcase uses calibrated directional gates / zones and conservative event logic. A real camera must be calibrated for its own doorway geometry.

## Existing hardware first

TrackBus follows an **existing-infrastructure-first** approach:

1. If usable APC already exists, integrate it.
2. If bus CCTV already exists, evaluate it before installing new cameras.
3. Calibrate suitable existing CCTV to the doorway geometry.
4. Recommend dedicated TrackBus cameras only when existing footage cannot provide acceptable results.

TrackBus should support reasonable overhead, angled and side-view camera positions through per-camera calibration. Software cannot recover information that was never captured, so extreme blur, permanent occlusion, very small subjects or poor night footage may still require better camera placement.

## Edge AI vs server AI

The pilot must determine where vision inference should run.

### On-bus / edge processing

```text
Camera → edge computer → IN/OUT event → small JSON → mobile network → server
```

Advantages: low bandwidth, raw video can stay local, counting can continue during network loss.

### Central-server processing

```text
Camera → mobile network → central GPU/server → Vision → events
```

Advantages: centralized compute and model updates, but much higher bandwidth and network dependency.

The correct architecture should be selected after measuring actual bus LTE/4G/5G coverage, upload stability, latency, dead zones and hardware requirements.

## Forecasting

TrackBus already contains a **transparent baseline forecast**. It is not yet a trained production ML demand model.

The current baseline uses recent occupancy and contextual factors such as time of day, weekday/weekend, weather and nearby events. Future ML models should be trained on real operational data and promoted only if they measurably outperform the baseline.

A useful passenger-facing result is therefore not only:

> `Occupancy: 72%`

but eventually:

> **Moderate now — expected to be crowded when it reaches your stop.**

## Passenger beta

The passenger interface translates occupancy into simple crowding levels:

- 🟢 Plenty of space
- 🟢 Seats likely available
- 🟡 Moderate
- 🟠 Crowded
- 🔴 Very crowded / Full

Passengers can also provide optional one-tap crowd reports. Community reports are a supporting signal and should never allow one user to overwrite automated occupancy by themselves.

## Validation

TrackBus separates three different problems:

1. **Detection:** did YOLO see the person?
2. **Tracking:** did the same person keep a consistent temporary track?
3. **Counting:** did the system emit the correct IN/OUT event?

Validation should use:

- manually annotated video events;
- controlled boarding/alighting tests;
- existing APC comparison where available;
- end-of-route / verified-empty reconciliation;
- day, night, crowded and normal-condition evaluation;
- optional driver crowd-level reference labels while stationary.

A detector metric such as mAP must not be presented as passenger-counting accuracy.

## Privacy and safety

TrackBus is designed for anonymous passenger flow:

- no facial recognition;
- no biometric identification;
- no cross-journey re-identification;
- temporary video-local IDs only;
- raw video stays on the edge by default for edge inference;
- the backend focuses on counts, events and system health;
- TrackBus does not control brakes, steering or other safety-critical vehicle systems.

## Next pilot step

### If existing footage is available

Request a small privacy-approved sample of representative APC/CCTV data, calibrate it, create manual ground truth, benchmark the current system and determine whether the existing hardware is sufficient.

### If suitable footage is not available

Run a small controlled pilot on roughly two buses to collect representative footage and measure real network conditions. Use that evidence to decide camera placement, edge-vs-server inference and final hardware requirements.

A working two-bus technical pilot has been estimated at roughly **50–80 million UZS**, with about **70 million UZS** as a planning figure. This is an internal estimate, not a vendor quote.

## Repository map

```text
app/                       Web showcase and gateway API
components/                Operator dashboard and passenger beta
contracts/                 Canonical passenger-count event schema
services/analytics-api/    Durable ingestion, quality and operations API
trackbus/                   Vision detection, tracking, counting and evaluation
showcase.py                 Executable webcam/video demonstration
configs/                    Camera, tracking and experiment configuration
tools/                      Replay and support utilities
docs/                       Architecture, pilot, privacy and evaluation guides
```

## Branch guide

| Branch | Purpose | Status |
| --- | --- | --- |
| [`main`](https://github.com/favi2202/trackbus-ai/tree/main) | Integrated TrackBus MVP: web app, API, Vision, forecasting, passenger beta, tests and docs | **Primary branch / start here** |
| [`agent/trackbus-foundation`](https://github.com/favi2202/trackbus-ai/tree/agent/trackbus-foundation) | Development history of the integrated platform foundation that was merged into `main` | Historical / reference |
| [`feature/trackbus-v0.2.1-event-stability`](https://github.com/favi2202/trackbus-ai/tree/feature/trackbus-v0.2.1-event-stability) | Focused CV event-stability, calibration and evaluation research | Research branch |
| [`feature/trackbus-v0.2-multiview`](https://github.com/favi2202/trackbus-ai/tree/feature/trackbus-v0.2-multiview) | Multi-view detection/fusion experiment | Research branch |
| [`feature/trackbus-v0.1.2`](https://github.com/favi2202/trackbus-ai/tree/feature/trackbus-v0.1.2) | Earlier single-camera YOLO + ByteTrack prototype | Historical CV baseline |
| [`develop`](https://github.com/favi2202/trackbus-ai/tree/develop) | Original project scaffold and early roadmap | Historical |

The repository's verified initial commit is dated **2026-07-01 07:30:04 UTC** (`b6e2232...`).

## Quick start

### Web

```bash
npm ci
npm run dev
```

### Python Vision

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
python showcase.py --camera 0
```

### Verification

```bash
npm run lint
npm test
python -m pytest -q
```

See [`docs/architecture.md`](docs/architecture.md), [`docs/showcase.md`](docs/showcase.md), [`docs/privacy-and-safety.md`](docs/privacy-and-safety.md), [`docs/detection-benchmark.md`](docs/detection-benchmark.md) and [`docs/pilot-runbook.md`](docs/pilot-runbook.md) for details.

---

# O‘zbekcha

## TrackBus nima?

**TrackBus — jamoat transporti uchun AI asosidagi transport-intelligence platformasi.** U operatorlarga yo‘lovchi oqimi, avtobus bandligi, talab va transport taqsimotidagi nomutanosiblikni tushunishga yordam berish uchun yaratilmoqda.

TrackBus faqat kamera bilan odam sanash tizimi emas. Platforma mavjud **APC, avtobus CCTV kameralari, GPS va tarixiy operatsion ma'lumotlardan** foydalanishi va TrackBus Vision'ni faqat kerak bo‘lgan joyda qo‘shishi mumkin.

> **Hozirgi holat:** MVP / tadqiqot prototipi. Integratsiyalashgan dashboard, API/backend asoslari, forecast baseline, passenger beta va computer-vision pipeline `main` branch ichida mavjud. TrackBus hali real Toshkent avtobuslarida production darajasida validatsiya qilinmagan va 90%+ aniqlik da'vosi qilinmaydi.

## Muammo

Bir avtobus haddan tashqari band bo‘lib, uning ortidan kelayotgan avtobus deyarli bo‘sh bo‘lishi mumkin. Boshqa qatnovlar esa yo‘nalishning katta qismida juda kam yo‘lovchi bilan yurishi mumkin.

Operatorlarda APC, CCTV, GPS yoki to‘lov tizimlaridan ma'lumot bo‘lishi mumkin, lekin ular alohida tizimlarda qolib ketishi mumkin. TrackBusning vazifasi — bu ma'lumotlarni operatsion qaror uchun foydali ko‘rinishga aylantirish.

## TrackBus Vision qanday ishlaydi?

```text
Kamera
  ↓
YOLO — odamni aniqlash
  ↓
ByteTrack — vaqtinchalik anonim ID
  ↓
kalibrlangan yo‘nalish / trajectory
  ↓
IN yoki OUT hodisasi
  ↓
JSON → API → database → dashboard
```

Yuzni tanish yo‘q. Doimiy yo‘lovchi identifikatori yo‘q.

## Mavjud uskunadan foydalanish birinchi o‘rinda

TrackBusning pilot tamoyili:

1. APC mavjud bo‘lsa — avval uni ishlatish.
2. CCTV mavjud bo‘lsa — avval o‘sha kamerani tekshirish.
3. Mos kamera bo‘lsa — uni kalibrlash va qayta ishlatish.
4. Faqat mavjud kamera yetarli bo‘lmasa — yangi kamera taklif qilish.

Kamera aynan eshikning tepasida bo‘lishi shart emas. Turli burchaklar uchun alohida kalibrovka qilinishi mumkin.

## Edge yoki server?

Agar avtobusdagi mobil internet zaif bo‘lsa, eng mantiqiy arxitektura:

```text
Kamera → avtobusdagi edge AI → sanash → kichik JSON → server
```

Internet vaqtincha uzilsa ham sanash davom etadi va eventlar keyin sinxronlanadi.

Agar tarmoq yetarlicha kuchli bo‘lsa, markaziy serverda video qayta ishlash varianti ham sinovdan o‘tkazilishi mumkin. Yakuniy qaror real avtobus tarmog‘i o‘lchangandan keyin qabul qilinadi.

## Forecasting

Hozir TrackBusda ishlaydigan **oddiy va tushunarli baseline forecast** mavjud. Bu hali real transport ma'lumotlarida o‘qitilgan production ML model emas.

Kelajakda real tarixiy ma'lumotlar orqali ML model o‘qitiladi va faqat baseline'dan yaxshiroq natija bersa ishlatiladi.

## Yo‘lovchi interfeysi

Oddiy `73%` ko‘rsatish o‘rniga:

- 🟢 Bo‘sh joy ko‘p
- 🟢 O‘rindiq topilishi mumkin
- 🟡 O‘rtacha
- 🟠 Tiqilinch
- 🔴 Juda tiqilinch / To‘la

Kelajakdagi muhim funksiya:

> **Hozir o‘rtacha — sizning bekatingizga kelganda tiqilinch bo‘lishi kutilmoqda.**

## Validatsiya

TrackBus alohida o‘lchaydi:

- YOLO odamni ko‘rdimi?
- tracker IDni saqladimi?
- yakuniy IN/OUT hodisasi to‘g‘rimi?

Aniqlik manual belgilangan video, boshqariladigan sinovlar va mavjud APC ma'lumotlari bilan tekshiriladi. Driver crowd labels ham qo‘shimcha reference signal bo‘lishi mumkin.

## Maxfiylik va xavfsizlik

- yuzni tanish yo‘q;
- biometrik identifikatsiya yo‘q;
- safarlar orasida odamni kuzatish yo‘q;
- faqat vaqtinchalik anonim track ID;
- edge ishlatilganda raw video odatda avtobusning o‘zida qoladi;
- TrackBus tormoz, rul yoki boshqa safety-critical tizimlarni boshqarmaydi.

## Keyingi bosqich

**Agar operatorlarda mavjud kamera yozuvlari bo‘lsa:** kichik sample olish, kalibrlash, manual ground truth yaratish va mavjud kameralar yetarliligini o‘lchash.

**Agar mos video bo‘lmasa:** taxminan 2 ta avtobusda kichik controlled pilot orqali kunduz/tun, bo‘sh/tiqilinch holatlar va mobil tarmoq sifatini yig‘ish.

Asosiy maqsad hozir city-wide deployment emas. Asosiy maqsad — **real ma'lumot bilan TrackBusni isbotlash va eng to‘g‘ri texnik arxitekturani tanlash**.

## Branchlar

| Branch | Vazifasi | Holati |
| --- | --- | --- |
| `main` | Integratsiyalashgan TrackBus MVP | **Asosiy branch** |
| `agent/trackbus-foundation` | `main`ga merge qilingan platforma foundation development tarixi | Reference / tarixiy |
| `feature/trackbus-v0.2.1-event-stability` | CV event stability va calibration/evaluation tadqiqoti | Research |
| `feature/trackbus-v0.2-multiview` | Multi-view detection eksperimenti | Research |
| `feature/trackbus-v0.1.2` | Oldingi YOLO + ByteTrack prototipi | Tarixiy |
| `develop` | Dastlabki scaffold va roadmap | Tarixiy |

**Start here:** `main`.
