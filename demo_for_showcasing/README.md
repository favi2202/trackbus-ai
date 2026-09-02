# Demo for showcasing

This directory labels the presentation layer and its synthetic scenarios clearly.
The runnable dashboard lives at the repository root because that is the hosted
application entry point. It never claims that its values are live government data.

## What the demonstration proves

1. A bus counter produces anonymous boarding and alighting events.
2. The platform validates the events and calculates current occupancy.
3. A transparent forecasting baseline predicts demand.
4. The operator dashboard recommends a dispatch action.
5. The public view shows understandable crowding bands.

The UI uses `data/sample/passenger-count-events.json`. Production integrations
will send the same contract to the analytics API.
