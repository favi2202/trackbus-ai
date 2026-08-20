# Privacy and operational safety

TrackBus optimizes transport capacity without identifying passengers.

- APC is the first-choice source when an approved counter already exists;
  TrackBus Vision is optional.
- Vision uses temporary process-local track IDs only. There is no face
  recognition, biometric matching, cross-camera identity, or passenger profile.
- Raw video remains on the edge by default. Approved validation footage requires
  documented purpose, access, retention, and deletion.
- The cloud contract contains counts, operational IDs, time, confidence, and
  quality flags—not frames or identity.
- Source data and API traffic require encryption, rotated device credentials,
  least-privilege access, and audit logs before production.
- Passengers receive occupancy bands and update confidence, not individual
  movement histories.
- Recommendations remain advisory and auditable. A trained operator approves
  operational changes; TrackBus does not control safety-critical vehicle systems.

Accuracy, ROI, and public-safety claims require representative pilot evidence.
Synthetic showcase values never satisfy that gate.
