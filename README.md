# TrackBus AI

**AI-powered transport intelligence for public transportation**

TrackBus is an MVP-stage transport-intelligence platform designed to help public transport operators understand **passenger flow, vehicle occupancy, crowding patterns, demand and operational imbalance**.

The project is not limited to a single camera system. TrackBus is designed to work with the infrastructure an operator already has — such as **APC sensors, existing bus CCTV, GPS and historical operational data** — and add TrackBus Vision only where it is useful.

> **Current status:** MVP / prototype. The integrated platform, web dashboard, API foundation, forecasting baseline and computer-vision pipeline exist, but TrackBus has **not yet been validated as a production passenger-counting system on representative Tashkent bus footage**. Forecast and dispatch scenarios in the showcase are clearly labeled synthetic. No 90%+ production accuracy claim is made.

**Languages:** [English](#english) · [O‘zbekcha](#ozbekcha)

---

# English

## 1. What problem does TrackBus solve?

Public transport demand and fleet supply do not always match.

A bus may run with very low occupancy for a large part of its route while another bus on the same corridor becomes overcrowded. Operators may already collect useful information through payment systems, GPS, APC devices or CCTV, but these sources can be fragmented and do not automatically become one clear operational picture.

TrackBus aims to answer questions such as:

- How full is each bus now?
- Where do passengers board and leave?
- Which route segments repeatedly operate far below capacity?
- Where and when does overcrowding happen?
- Are buses on the same route badly spaced, with one overloaded and the next mostly empty?
- What crowding level is likely when a bus reaches the passenger’s stop?
- Which operational changes should be investigated: timing, frequency, vehicle size, dispatching or route allocation?

The goal is **not simply to count people**. The goal is to turn passenger-flow data into transport decisions.

---

## 2. The TrackBus approach

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

Payment history can also be used as an **additional historical demand signal** when access is available. Real-time payment integration is a later integration opportunity and is not a requirement for the current MVP.

---

## 3. What exists today?

The current integrated MVP is on the [`agent/trackbus-foundation`](https://github.com/favi2202/trackbus-ai/tree/agent/trackbus-foundation) branch.

It currently includes:

- an operator / Live Pilot dashboard;
- current vehicle occupancy and source-health views;
- canonical passenger-count JSON event contracts;
- durable, idempotent analytics/API ingestion;
- a Python TrackBus Vision pipeline based on Ultralytics YOLO + ByteTrack;
- configurable camera calibration and crossing zones;
- a conservative `OUTSIDE -> DOOR -> INSIDE` / reverse transition model;
- webcam, local-video and demo showcase modes;
- offline event queuing when API connectivity is unavailable;
- a transparent baseline occupancy forecasting engine;
- passenger-facing crowding / forecast demonstration views;
- replay, quality, forecasting, API, web and vision tests;
- architecture, pilot, privacy and deployment documentation.

### Hosted showcase

The integrated branch documents a hosted MVP showcase at:

**https://trackbus-showcase.favi-2202.chatgpt.site/**

The showcase is a prototype. Synthetic forecast/dispatch scenarios are labeled as such.

---

## 4. TrackBus Vision: how passenger counting works

TrackBus Vision does **person detection and short-term anonymous tracking**. It does not perform facial recognition.

```text
Camera frame
   ↓
YOLO person detection
   ↓
ByteTrack temporary track IDs
   ↓
trajectory / calibrated zones
   ↓
OUTSIDE -> DOOR -> INSIDE  = IN
INSIDE  -> DOOR -> OUTSIDE = OUT
   ↓
normalized passenger-count event
   ↓
API / database / dashboard
```

Temporary IDs exist only to follow motion inside the local processing session. They are not passenger identities.

A typical event concept is:

```json
{
  "source": "vision",
  "bus_id": "BUS-014",
  "route_id": "22",
  "door_id": "front",
  "boardings": 1,
  "alightings": 0,
  "occupancy": 37,
  "quality_flags": []
}
```

The production goal is to transmit small operational events rather than continuously sending raw video whenever edge processing is used.

---

## 5. Existing hardware first

TrackBus is designed around an **existing-infrastructure-first** principle.

Before proposing new cameras, the pilot should ask:

1. Is APC already installed and accessible?
2. Are usable bus CCTV recordings or streams already available?
3. Does the existing camera see the doorway clearly enough?
4. Is resolution, frame rate, night quality and passenger size sufficient?
5. Can the existing camera be calibrated for reliable passenger trajectories?

If existing CCTV is suitable, TrackBus should use it. A dedicated camera is only justified where existing footage cannot achieve acceptable counting quality.

### Different camera angles

TrackBus should not require every camera to be directly above the door.

Per-camera calibration can define areas such as:

```text
OUTSIDE  ->  DOOR / TRANSITION  ->  INSIDE
```

Different buses can therefore use different polygon geometry, orientation, exclusions and — where useful — perspective calibration.

Software cannot recover information that a camera never captured, so extremely poor visibility, severe blur, permanent occlusion or tiny subjects may still require a better camera position.

---

## 6. Edge AI vs central-server AI

One of the most important pilot questions is **where the vision model should run**.

### Option A — Process on the bus

```text
Camera -> edge AI computer -> IN/OUT JSON events -> mobile network -> TrackBus server
```

Advantages:

- very low mobile-data usage;
- raw video can remain local by default;
- counting can continue during temporary network loss;
- only small events need synchronization.

Trade-off: each equipped bus needs sufficient onboard compute, storage, power protection and installation hardware.

### Option B — Send video to a central server

```text
Camera -> mobile network -> central GPU/server -> TrackBus Vision -> events
```

Advantages:

- less AI compute hardware on each bus;
- centralized model updates and maintenance.

Trade-offs:

- much higher bandwidth;
- stronger dependence on mobile-network quality;
- larger server/GPU requirements;
- greater privacy/security and video-retention considerations.

### How TrackBus should decide

During a pilot we should measure real bus connectivity:

- LTE/4G/5G availability;
- upload speed and stability while moving;
- latency and dead zones;
- realistic video bandwidth;
- local hardware requirements and cost.

The final architecture should be selected from **measured bus conditions**, not assumptions.

---

## 7. Forecasting

TrackBus already contains a working **transparent baseline forecast**, not a trained production ML demand model.

The current baseline uses recent occupancy plus contextual factors such as time of day, weekday/weekend, weather and nearby events to produce an expected occupancy range.

The long-term plan is:

```text
real historical APC / passenger-flow data
              ↓
       train candidate models
              ↓
 compare against transparent baseline
              ↓
 only promote ML if it performs better
```

A future passenger-facing output can therefore be more useful than a raw percentage:

> **Moderate now — expected to be crowded when it reaches your stop.**

---

## 8. Passenger experience

Instead of forcing passengers to interpret numbers such as `73% occupancy`, TrackBus can translate occupancy into simple crowding levels:

- 🟢 **Plenty of space**
- 🟢 **Seats likely available**
- 🟡 **Moderate**
- 🟠 **Crowded**
- 🔴 **Very crowded / Full**

A passenger could compare two approaching buses and choose whether waiting a few more minutes is likely to provide a more comfortable ride.

Optional community reports can later provide an additional signal, but one passenger report should never override automated data by itself.

---

## 9. Validation: how do we know the system is correct?

TrackBus separates **detection quality, tracking quality and final counting quality**.

The pilot should use multiple validation methods:

- manually annotated video events;
- controlled boarding / alighting tests;
- comparison with existing APC where available;
- end-of-route occupancy reconciliation;
- selected low-confidence event review;
- day / night / crowded / normal-condition evaluation;
- optional driver reference labels while the bus is stationary.

A driver reference interface could use four simple labels:

- Almost empty
- Moderate
- Crowded
- Very crowded

Driver labels are useful repeated human observations, but they are **reference labels, not perfect ground truth**.

TrackBus should never call detector mAP “passenger-counting accuracy.” Final accuracy must be measured against independently annotated passenger events.

---

## 10. Privacy and safety

TrackBus Vision is designed for **anonymous passenger flow**, not passenger identity.

Current principles:

- no facial recognition;
- no biometric identification;
- no cross-journey passenger re-identification;
- temporary video-local track IDs only;
- raw video stays on the edge by default when edge inference is used;
- server-side data focuses on count/events and system health;
- TrackBus does not control brakes, steering or safety-critical vehicle systems;
- operator recommendations remain recommendations — the operator makes the final decision.

A real deployment would still require appropriate legal review, retention rules, access control, cybersecurity and passenger-information policies.

---

## 11. Proposed next step

### Scenario A — existing footage is available

If the transport operator already has suitable CCTV recordings:

1. obtain a representative privacy-approved sample;
2. evaluate the current detector and tracker;
3. calibrate existing camera angles;
4. manually annotate representative passenger crossings;
5. measure detection, tracking and event-level errors;
6. fine-tune the model only where evidence shows it is necessary;
7. decide whether existing cameras are sufficient.

This is the lowest-cost validation path.

### Scenario B — suitable footage is not available

Use a small controlled pilot on approximately two buses to collect representative data:

- daytime and night;
- quiet and crowded periods;
- different doors / camera angles;
- simultaneous boarding and alighting;
- motion blur, occlusion and edge cases;
- mobile-network measurements.

The pilot then determines the real camera, compute, connectivity and deployment requirements.

---

## 12. Pilot budget direction

A small two-bus technical pilot is currently planned in the approximate **50–80 million UZS** range, with roughly **70 million UZS** as a working planning figure.

Typical budget categories include:

- cameras and mounting, only where existing CCTV is insufficient;
- edge AI computers if onboard processing is selected;
- SSD/local storage;
- LTE/4G connectivity hardware;
- protected power conversion and installation;
- cabling/enclosures;
- data annotation and AI improvement;
- backend/database/monitoring;
- validation and field testing;
- contingency for real installation conditions.

This is a planning estimate, not a vendor quote. Final cost depends strongly on how much existing infrastructure can be reused and whether inference runs onboard or centrally.

---

## 13. Repository / branch guide

| Branch | Purpose | Status |
| --- | --- | --- |
| [`main`](https://github.com/favi2202/trackbus-ai/tree/main) | Project landing page and reviewer entry point | Current public overview |
| [`agent/trackbus-foundation`](https://github.com/favi2202/trackbus-ai/tree/agent/trackbus-foundation) | Integrated MVP: dashboard, API, database foundation, Vision, showcase, forecasting, docs | **Primary integrated MVP branch** |
| [`feature/trackbus-v0.2.1-event-stability`](https://github.com/favi2202/trackbus-ai/tree/feature/trackbus-v0.2.1-event-stability) | Computer-vision event stability, camera calibration, manual ground truth and evaluation | Latest focused CV stability branch |
| [`feature/trackbus-v0.2-multiview`](https://github.com/favi2202/trackbus-ai/tree/feature/trackbus-v0.2-multiview) | Separated detection/tracking and multi-view inference experiments | Research / experiment branch |
| [`feature/trackbus-v0.1.2`](https://github.com/favi2202/trackbus-ai/tree/feature/trackbus-v0.1.2) | Single-video YOLO + ByteTrack baseline with ROI, lanes, exclusions and diagnostics | Earlier CV baseline |
| [`develop`](https://github.com/favi2202/trackbus-ai/tree/develop) | Original early project scaffold / roadmap | Historical |

### Recommended path for reviewers

If you are reviewing TrackBus for a startup program, grant or technical pilot:

1. Start with this README for the product and pilot strategy.
2. Open [`agent/trackbus-foundation`](https://github.com/favi2202/trackbus-ai/tree/agent/trackbus-foundation) for the integrated MVP.
3. Use [`feature/trackbus-v0.2.1-event-stability`](https://github.com/favi2202/trackbus-ai/tree/feature/trackbus-v0.2.1-event-stability) for the focused passenger-counting research history and evaluation tools.

---

## 14. Running the integrated MVP

From `agent/trackbus-foundation`:

### Web showcase

```bash
npm ci
npm run dev
```

### Vision showcase

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'

python showcase.py --camera 0
python showcase.py --video path/to/video.mp4
python showcase.py --demo
python showcase.py --calibrate --camera 0
```

For detailed setup, testing and deployment instructions, use the README and `docs/` directory on the integrated branch.

---

## 15. What TrackBus is — and what it is not

### TrackBus is

- an MVP-stage transport-intelligence platform;
- a vendor-neutral data and analytics layer;
- an optional computer-vision passenger-counting system;
- an operator dashboard and passenger-information concept;
- an evidence-driven pilot project.

### TrackBus is not yet

- a production-certified passenger-counting system;
- a validated 90%+ accuracy claim across real Tashkent buses;
- an official city-wide deployment;
- a facial-recognition system;
- a replacement for every existing APC/CCTV/payment system.

The next milestone is **real-world validation with representative operator data or a small controlled bus pilot**.

---

# O‘zbekcha

## 1. TrackBus qanday muammoni hal qiladi?

Jamoat transportida yo‘lovchi talabi va avtobuslar taqsimoti har doim ham bir-biriga mos kelmaydi.

Ba’zi avtobuslar yo‘nalishning katta qismida juda kam yo‘lovchi bilan yurishi mumkin, boshqa avtobuslar esa ayni yo‘nalishda haddan tashqari band bo‘lishi mumkin. Operatorlarda to‘lov tizimi, GPS, APC yoki CCTV orqali foydali ma’lumotlar mavjud bo‘lishi mumkin, ammo bu ma’lumotlar ko‘pincha alohida tizimlarda saqlanadi va avtomatik ravishda yagona operatsion tasvirga aylanmaydi.

TrackBus quyidagi savollarga javob berishga yordam berishni maqsad qiladi:

- Hozir har bir avtobus qanchalik band?
- Yo‘lovchilar qayerda chiqadi va qayerda tushadi?
- Qaysi yo‘nalish qismlari muntazam ravishda past yuklama bilan ishlaydi?
- Qayerda va qachon haddan tashqari bandlik yuz beradi?
- Bir yo‘nalishdagi avtobuslar noto‘g‘ri intervalda yurib, biri to‘lib, keyingisi deyarli bo‘sh ketmayaptimi?
- Avtobus yo‘lovchining bekatiga yetib kelganda qanchalik band bo‘lishi mumkin?
- Qaysi operatsion o‘zgarishlarni tekshirish kerak: interval, chastota, avtobus hajmi, qo‘shimcha avtobus yoki yo‘nalish taqsimoti?

Maqsad faqat **odam sanash emas**. Maqsad — yo‘lovchi oqimi ma’lumotlarini transport qarorlariga aylantirish.

---

## 2. TrackBus yondashuvi

```text
Mavjud APC / CCTV / TrackBus Vision / GPS / tarixiy ma’lumotlar
                              ↓
                    standartlashtirilgan eventlar
                              ↓
                         TrackBus API
                              ↓
                 bandlik + sifat tekshiruvi
                              ↓
                 tahlil + forecast baseline
                              ↓
             operator dashboardi + yo‘lovchi interfeysi
```

To‘lov tarixidan ham ruxsat mavjud bo‘lganda **qo‘shimcha tarixiy talab signali** sifatida foydalanish mumkin. Real-time to‘lov integratsiyasi keyingi bosqich imkoniyati bo‘lib, hozirgi MVP uchun majburiy emas.

---

## 3. Hozir nimalar mavjud?

Hozirgi integratsiyalangan MVP [`agent/trackbus-foundation`](https://github.com/favi2202/trackbus-ai/tree/agent/trackbus-foundation) branchida joylashgan.

Unda quyidagilar mavjud:

- operator / Live Pilot dashboardi;
- avtobus bandligi va data-source health ko‘rsatkichlari;
- standart passenger-count JSON event contractlari;
- idempotent analytics/API ingestion;
- Ultralytics YOLO + ByteTrack asosidagi TrackBus Vision pipeline;
- kamera kalibrovkasi va crossing zonalari;
- konservativ `OUTSIDE -> DOOR -> INSIDE` va teskari o‘tish modeli;
- webcam, video va demo showcase rejimlari;
- internet bo‘lmaganda eventlarni lokal navbatda saqlash;
- tushunarli baseline occupancy forecasting engine;
- yo‘lovchi uchun bandlik/forecast demo interfeysi;
- API, web, forecast, quality va vision testlari;
- arxitektura, pilot, privacy va deployment hujjatlari.

### Hosted showcase

Integratsiyalangan branchda quyidagi MVP showcase ko‘rsatilgan:

**https://trackbus-showcase.favi-2202.chatgpt.site/**

Bu prototip. Synthetic forecast va dispatch ssenariylari tegishli ravishda belgilangan.

---

## 4. TrackBus Vision qanday ishlaydi?

TrackBus Vision **odamni aniqlash va qisqa muddatli anonim tracking** qiladi. Face recognition ishlatilmaydi.

```text
Kamera kadri
   ↓
YOLO person detection
   ↓
ByteTrack vaqtinchalik track ID
   ↓
trajectory / kalibrlangan zonalar
   ↓
OUTSIDE -> DOOR -> INSIDE  = IN
INSIDE  -> DOOR -> OUTSIDE = OUT
   ↓
standart passenger-count event
   ↓
API / database / dashboard
```

Vaqtinchalik ID faqat lokal video processing davomida harakatni kuzatish uchun ishlatiladi. U yo‘lovchining shaxsini aniqlamaydi.

Event misoli:

```json
{
  "source": "vision",
  "bus_id": "BUS-014",
  "route_id": "22",
  "door_id": "front",
  "boardings": 1,
  "alightings": 0,
  "occupancy": 37,
  "quality_flags": []
}
```

Edge processing ishlatilganda production maqsadi — raw videoni doimiy yuborish o‘rniga kichik operatsion eventlarni yuborish.

---

## 5. Avval mavjud uskunadan foydalanish

TrackBus **existing-infrastructure-first** prinsipiga asoslanadi.

Yangi kamera taklif qilishdan oldin pilot quyidagilarni tekshiradi:

1. APC allaqachon o‘rnatilganmi va ma’lumotga ruxsat bormi?
2. Avtobuslarda foydalanish mumkin bo‘lgan CCTV video yoki stream mavjudmi?
3. Mavjud kamera eshik zonasini yetarlicha ko‘radimi?
4. Resolution, FPS, tungi sifat va odamlarning kadrdagi o‘lchami yetarlimi?
5. Mavjud kamera yo‘lovchi trajectory uchun kalibrlanishi mumkinmi?

Agar mavjud CCTV yetarli bo‘lsa, TrackBus aynan undan foydalanishi kerak. Maxsus yangi kamera faqat mavjud video yetarli sifat bermasa kerak bo‘ladi.

### Turli kamera burchaklari

TrackBus har bir kamerani aynan eshik tepasida bo‘lishini talab qilmasligi kerak.

Har bir kamera uchun alohida kalibrovka orqali quyidagi zonalar belgilanadi:

```text
OUTSIDE  ->  DOOR / TRANSITION  ->  INSIDE
```

Shuning uchun turli avtobuslarda turli polygon, orientation, exclusion va kerak bo‘lsa perspective calibration ishlatilishi mumkin.

Lekin kamera umuman kerakli ma’lumotni yozib olmagan bo‘lsa, dasturiy ta’minot uni “yarata” olmaydi. Juda yomon ko‘rinish, kuchli blur, doimiy to‘siq yoki juda kichik odam tasviri bo‘lsa, kamera pozitsiyasini yaxshilash kerak bo‘lishi mumkin.

---

## 6. Edge AI yoki markaziy server AI?

Pilotdagi muhim savollardan biri — **vision model qayerda ishlashi kerak?**

### Variant A — avtobus ichida processing

```text
Kamera -> edge AI kompyuter -> IN/OUT JSON event -> mobil tarmoq -> TrackBus server
```

Afzalliklari:

- mobil internet sarfi juda kam;
- raw video odatda avtobus ichida qoladi;
- internet vaqtincha uzilsa ham sanash davom etishi mumkin;
- keyinchalik faqat kichik eventlar sync qilinadi.

Kamchiligi: har bir avtobusda yetarli hisoblash qurilmasi, storage, power protection va o‘rnatish uskunasi kerak bo‘ladi.

### Variant B — videoni markaziy serverga yuborish

```text
Kamera -> mobil tarmoq -> markaziy GPU/server -> TrackBus Vision -> eventlar
```

Afzalliklari:

- har bir avtobusda kuchli AI kompyuter kamroq kerak bo‘ladi;
- modelni markazdan yangilash osonroq.

Kamchiliklari:

- ancha katta bandwidth;
- mobil tarmoq sifatiga kuchli bog‘liqlik;
- kuchliroq server/GPU talabi;
- privacy, security va video retention talablari kattaroq.

### Qanday tanlanadi?

Pilot vaqtida real avtobus sharoitida quyidagilar o‘lchanadi:

- LTE/4G/5G mavjudligi;
- avtobus harakatda bo‘lganda upload sifati;
- latency va dead zone;
- real video bandwidth;
- lokal hardware narxi va talabi.

Yakuniy arxitektura taxmin bilan emas, **real avtobus o‘lchovlari** asosida tanlanishi kerak.

---

## 7. Forecasting

TrackBusda hozir ishlaydigan **tushunarli baseline forecast** mavjud. Bu hali real transport datasetida train qilingan production ML demand modeli emas.

Hozirgi baseline recent occupancy va vaqt, weekday/weekend, ob-havo va yaqin event kabi faktorlar orqali expected occupancy diapazonini hisoblaydi.

Keyingi maqsad:

```text
real tarixiy APC / passenger-flow data
                 ↓
          candidate model train
                 ↓
       baseline bilan taqqoslash
                 ↓
  faqat yaxshiroq bo‘lsa ML modelni qo‘llash
```

Kelajakda yo‘lovchi uchun oddiy foizdan ko‘ra foydaliroq xabar berish mumkin:

> **Hozir o‘rtacha — sizning bekatingizga kelganda tiqilinch bo‘lishi kutilmoqda.**

---

## 8. Yo‘lovchi interfeysi

Yo‘lovchiga `73% bandlik` kabi raqamni tushunishga majbur qilish o‘rniga TrackBus bandlikni oddiy holatlarga aylantirishi mumkin:

- 🟢 **Bo‘sh joy ko‘p**
- 🟢 **O‘rindiq topish ehtimoli yuqori**
- 🟡 **O‘rtacha**
- 🟠 **Tiqilinch**
- 🔴 **Juda tiqilinch / To‘la**

Yo‘lovchi bir yo‘nalishda kelayotgan ikki avtobusni solishtirib, biroz kutsa qulayroq avtobus kelishi mumkinligini ko‘rishi mumkin.

Kelajakda passenger community report qo‘shimcha signal bo‘lishi mumkin, lekin bitta yo‘lovchi reporti avtomatik ma’lumotni yolg‘iz o‘zi almashtirmasligi kerak.

---

## 9. Validation: TrackBus to‘g‘ri sanayotganini qanday bilamiz?

TrackBus **detection sifati, tracking sifati va yakuniy counting sifatini** alohida o‘lchaydi.

Pilotda bir nechta validation usullari ishlatilishi kerak:

- odam tomonidan annotation qilingan video eventlar;
- nazorat qilinadigan kirish/chiqish testlari;
- mavjud APC bilan taqqoslash;
- route oxirida occupancy reconciliation;
- low-confidence eventlarni tanlab ko‘rib chiqish;
- kunduz / tun / crowded / normal holatlarni alohida baholash;
- avtobus to‘xtab turgan paytda optional driver reference label.

Driver reference uchun to‘rtta oddiy variant bo‘lishi mumkin:

- Deyarli bo‘sh
- O‘rtacha
- Tiqilinch
- Juda tiqilinch

Driver label foydali inson kuzatuvi, lekin **perfect ground truth emas**.

Detector mAP ko‘rsatkichi “passenger counting accuracy” deb atalmasligi kerak. Yakuniy aniqlik mustaqil annotation qilingan yo‘lovchi eventlari bilan o‘lchanadi.

---

## 10. Privacy va safety

TrackBus Vision **anonim yo‘lovchi oqimi** uchun mo‘ljallangan, odamning shaxsini aniqlash uchun emas.

Asosiy prinsiplar:

- face recognition yo‘q;
- biometrik identifikatsiya yo‘q;
- turli safarlar orasida passenger re-identification yo‘q;
- faqat vaqtinchalik lokal track ID;
- edge inference bo‘lsa raw video odatda lokal qoladi;
- serverga count/event va system-health ma’lumotlari yuboriladi;
- TrackBus brake, steering yoki safety-critical tizimlarni boshqarmaydi;
- operator tavsiyalari faqat tavsiya bo‘lib qoladi, yakuniy qarorni operator qiladi.

Real deployment uchun baribir legal review, retention policy, access control, cybersecurity va passenger-information qoidalari kerak bo‘ladi.

---

## 11. Taklif qilinayotgan keyingi qadam

### Scenario A — mavjud video bor

Agar transport operatorida mos CCTV video mavjud bo‘lsa:

1. representative va privacy-approved sample olish;
2. current detector/tracker ni test qilish;
3. mavjud kamera burchaklarini kalibrlash;
4. representative crossinglarni qo‘lda annotation qilish;
5. detection, tracking va event-level xatolarni o‘lchash;
6. faqat evidence ko‘rsatsa modelni fine-tune qilish;
7. mavjud kameralar yetarlimi-yo‘qmi aniqlash.

Bu eng arzon validation yo‘li.

### Scenario B — mos video yo‘q

Taxminan 2 ta avtobusda kichik controlled pilot orqali representative data yig‘ish:

- kunduz va tun;
- kam va ko‘p yo‘lovchili payt;
- turli eshik/kamera burchaklari;
- bir vaqtda kirish va chiqish;
- blur, occlusion va edge-case holatlar;
- mobil tarmoq o‘lchovlari.

Shundan keyin haqiqiy camera, compute, connectivity va deployment talablari aniqlanadi.

---

## 12. Pilot budjet yo‘nalishi

Kichik ikki avtobuslik texnik pilot uchun hozirgi rejalashtirish diapazoni taxminan **50–80 mln so‘m**, ishchi planning figure sifatida **~70 mln so‘m**.

Asosiy xarajat yo‘nalishlari:

- mavjud CCTV yetarli bo‘lmasa kamera va mounting;
- onboard processing tanlansa edge AI kompyuter;
- SSD/local storage;
- LTE/4G connectivity hardware;
- xavfsiz power conversion va installation;
- cable/enclosure;
- data annotation va AI improvement;
- backend/database/monitoring;
- validation va field testing;
- real installation sharoiti uchun contingency.

Bu vendor quote emas, rejalashtirish estimate. Yakuniy narx mavjud infratuzilmadan qancha foydalanish mumkinligi va inference avtobus ichida yoki markaziy serverda ishlashiga kuchli bog‘liq.

---

## 13. Repository / branch guide

| Branch | Vazifasi | Holati |
| --- | --- | --- |
| [`main`](https://github.com/favi2202/trackbus-ai/tree/main) | Loyiha landing page va reviewer uchun kirish nuqtasi | Hozirgi umumiy overview |
| [`agent/trackbus-foundation`](https://github.com/favi2202/trackbus-ai/tree/agent/trackbus-foundation) | Integratsiyalangan MVP: dashboard, API, database foundation, Vision, showcase, forecast, docs | **Asosiy integrated MVP branch** |
| [`feature/trackbus-v0.2.1-event-stability`](https://github.com/favi2202/trackbus-ai/tree/feature/trackbus-v0.2.1-event-stability) | Vision event stability, kamera calibration, manual ground truth va evaluation | Eng yangi focused CV stability branch |
| [`feature/trackbus-v0.2-multiview`](https://github.com/favi2202/trackbus-ai/tree/feature/trackbus-v0.2-multiview) | Detection/tracking separation va multi-view experimentlar | Research / experiment branch |
| [`feature/trackbus-v0.1.2`](https://github.com/favi2202/trackbus-ai/tree/feature/trackbus-v0.1.2) | Single-video YOLO + ByteTrack baseline, ROI, lanes, exclusions, diagnostics | Oldingi CV baseline |
| [`develop`](https://github.com/favi2202/trackbus-ai/tree/develop) | Dastlabki project scaffold / roadmap | Tarixiy |

### Reviewer uchun tavsiya etilgan yo‘l

Agar TrackBus startup program, grant yoki technical pilot uchun ko‘rib chiqilayotgan bo‘lsa:

1. Product va pilot strategiya uchun avval shu README ni o‘qing.
2. Integrated MVP uchun [`agent/trackbus-foundation`](https://github.com/favi2202/trackbus-ai/tree/agent/trackbus-foundation) branchini oching.
3. Passenger-counting research va evaluation tools uchun [`feature/trackbus-v0.2.1-event-stability`](https://github.com/favi2202/trackbus-ai/tree/feature/trackbus-v0.2.1-event-stability) branchidan foydalaning.

---

## 14. Integrated MVP ni ishga tushirish

`agent/trackbus-foundation` branchida:

### Web showcase

```bash
npm ci
npm run dev
```

### Vision showcase

Python 3.11+ tavsiya qilinadi.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'

python showcase.py --camera 0
python showcase.py --video path/to/video.mp4
python showcase.py --demo
python showcase.py --calibrate --camera 0
```

Batafsil setup, test va deployment ko‘rsatmalari uchun integrated branchdagi README va `docs/` papkasiga qarang.

---

## 15. TrackBus nima — va hozircha nima emas?

### TrackBus — bu

- MVP bosqichidagi transport-intelligence platforma;
- vendor-neutral data va analytics layer;
- optional computer-vision passenger-counting system;
- operator dashboard va passenger-information konsepti;
- evidence-driven pilot loyiha.

### TrackBus hozircha — bu emas

- production-certified passenger-counting tizimi;
- real Tashkent avtobuslarida validatsiya qilingan 90%+ accuracy claim;
- rasmiy city-wide deployment;
- face-recognition tizimi;
- barcha mavjud APC/CCTV/to‘lov tizimlarini almashtiruvchi mahsulot.

Keyingi asosiy milestone — **real operator data yoki kichik controlled bus pilot orqali real-world validation**.

---

## Project principle

> **Use existing infrastructure first. Measure before claiming. Add hardware only when evidence shows it is necessary.**

> **Avval mavjud infratuzilmadan foydalanish. Claim qilishdan oldin o‘lchash. Faqat evidence kerakligini ko‘rsatsa yangi hardware qo‘shish.**
