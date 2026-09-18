# Decision Log

This file records the main choices I made while shaping the project. The goal is to make the reasoning easy to discuss, challenge, and improve.

## 1. Use simulated data

Real international scouting and signing information is proprietary. I used simulated records so I could demonstrate the workflow without presenting invented information as fact or using confidential data.

## 2. Resolve identity before analyzing performance

If records are attached to the wrong player, every downstream result is compromised. The pipeline therefore matches on source player ID first. It uses normalized name and birth date only as a controlled fallback, and sends unresolved records to a review queue.

## 3. Preserve questionable records instead of silently fixing them

An automatic correction can hide the exact issue a scout or analyst needs to see. The project keeps the source values, records the validation rule and severity, and separates the row for review.

## 4. Hide very small comparison groups

The explorer suppresses groups with fewer than eight observations. Eight is not a universal baseball standard; it is a conservative prototype threshold meant to prevent a handful of simulated players from looking like a meaningful market pattern. With real data, I would set the threshold with analysts and decision-makers and test its sensitivity.

## 5. Show uncertainty, not only an outcome rate

Two groups can have the same observed rate and very different levels of evidence. Wilson intervals give the user a clearer view of uncertainty, especially when samples are modest.

## 6. Use a simple model and a time-based test

I used Ridge regression because it is transparent, regularized, and appropriate for a baseline demonstration. Training on earlier seasons and testing on a later season better reflects the real decision sequence than a random split. The model is not intended for actual player evaluation.

## 7. Design the output for a baseball user

The HTML explorer opens locally and does not require the user to write SQL. The filters mirror questions a staff member might ask, while the database and example queries remain available for deeper analysis.

## 8. Use AI coding tools openly

I defined the baseball problem, data structure, matching priorities, validation rules, and intended outputs. I used AI coding tools to help implement the prototype. I reviewed the workflow, ran the tests, and documented what the project can and cannot support. I would describe this accurately in an interview rather than claim production-level Python fluency I have not yet earned.
