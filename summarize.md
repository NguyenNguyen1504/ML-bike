# The whole project, in plain words

No jargon. If a word has to be technical, it is explained right there.

*Updated for Stage 2. The Stage 1 version of this story — where the model was not told
where anyone was going — is in git history.*

---

## 1. What are we trying to do?

You are standing at a city bike rack. Before you ride, you want to know:
**how long will this ride take?**

We guess that number before the ride starts, using three things you already know:

- **where you are going** — which station you take the bike from, and which one you
  will leave it at
- **when** — what day it is, what time it is
- **the weather** — warm or cold, dry or raining

## 2. What do we have to work with?

- **Every city bike ride** in Helsinki and Espoo for four summers (April–October, 2022
  to 2025). About **10 million rides**. Each one tells us when it started, which two
  stations it went between, how long it took, and how far it went.
- **The weather for every one of those days** — temperature and rain — from the Finnish
  weather service.

## 3. The rule we changed

In Stage 1, we did **not** tell the model where the rider was going. We thought that would
be cheating. It turns out we had mixed up two different things:

- **How far this exact ride went.** You only know this *after* the ride. Telling the model
  would be cheating, so it stays secret.
- **Which route you picked.** You know this *before* you start. Hiding it was like asking
  "how long does a trip take?" without saying where to.

So now the model knows the route. But it never sees this ride's own distance. Instead it
learns **how long that route usually is** from older rides — like a map that remembers.

How much did hiding the route cost us? Without it, the model beat "always say 11 minutes"
by about 1%. With it, it misses the typical ride by under 2 minutes instead of over 5.

## 4. What the notebook does, step by step

1. **Read** all 28 monthly files — 10.1 million rides.
2. **Throw away the broken ones:** bikes pulled out and pushed straight back, plus bikes
   being moved to the repair workshop (that is maintenance, not a ride). About 9.84
   million rides survive — 97 out of every 100.
3. **Look at the data.** A normal ride is about **11 minutes**, but a few forgotten bikes
   "ride" for days.
4. **Check our guesses** about what makes a ride slow (section 6 below).
5. **Learn every route's usual length** from older rides.
6. **Split by year, like exams.** Learn from 2022 and 2023. Check ourselves on 2024. Keep
   2025 locked in a drawer as the final exam.
7. **Teach three models**, next to two "no-brain" guessers for comparison.
8. **Check whether the differences are real**, not luck (section 7 explains how).
9. **Open the drawer** and take the final exam once.

There are two notebooks:

- `ML_bike.ipynb` — the Stage 2 story, 14 steps.
- `ML_bike_stage1/ML_bike_stage1.ipynb` — the Stage 1 version we already handed in. Frozen.

## 5. Forgotten bikes, and two tricks against them

Most rides are about 11 minutes. The longest "ride" in our data lasted **194 days**.
Nobody cycled for 194 days. Somebody forgot to put the bike back.

When you teach a model, you tell it how to count its mistakes. The usual way is called
**squared error**: being wrong by 2 counts as 4, being wrong by 10 counts as 100. Put a
194-day ride in front of that, and the model twists its whole answer to avoid being hugely
wrong about one bike:

> **0.3% of the rides cause 99.6% of the pain the model is trying to reduce.**

We use two tricks against this, and we checked that we need both:

- **The log trick.** Measure time in "how many times longer" instead of in minutes. A
  194-day ride becomes "very long" instead of "unimaginably long". That shrinks the
  forgotten bikes' share of the pain from 99.6% to 9%.
- **Counting mistakes kindly (Huber).** Big mistakes count gently, so the few forgotten
  bikes that are left cannot shout over everyone else.

With only the log trick, the model still guesses ordinary rides about **half a minute too
long**. Add Huber, and that goes away: on the final exam, **8 more rides out of every 100**
land within 2 minutes of the real time.

## 6. Our three guesses, checked

Before modelling, we guessed what sets a ride's length. Here is how each guess held up:

| Our guess | What the data says |
|---|---|
| A longer route takes longer | **Yes — and it is almost everything.** Route length does 99.8% of the work. |
| The time of day matters, because of traffic | **Yes, but only a little, and not because of traffic.** Morning commuters are the *fastest* riders of the whole day. Weekend rides are 4–5% slower. It is about *why* people ride, not about cars. |
| Bad weather makes riders slower | **No.** In the rain, people ride slightly *faster* and take *shorter* trips. Rain decides *who* rides, not how fast. |

Two fun facts the model found:

- **Doubling the route makes a ride 80% longer, not twice as long.** Long rides go faster —
  fewer traffic lights per kilometre.
- **Round trips** — returning the bike where you took it — **take about twice as long** as
  other rides of the same length. Those are leisure rides, with stops.

## 7. What came out (the final exam, 2025)

"Typical miss" means: for half the rides the guess is closer than this, for half it is
further off.

| How we guess | Typical miss | Rides within 2 min |
|---|---|---|
| Always say "11 minutes" | 5.4 min | 19 out of 100 |
| Physics: route length ÷ one usual speed (11.5 km/h), no learning | 2.1 min | 48 out of 100 |
| Model, counting mistakes the usual way (squared) | 2.4 min | 43 out of 100 |
| **Our model (Huber)** | **1.9 min** | **52 out of 100** |
| A fancier model (boosted trees) | 1.9 min | 52 out of 100 |

Read it as a ladder:

```
always say 11 minutes    5.4 min off
                           |  <-- a huge win, just from knowing the route
physics, no learning     2.1 min off
                           |  <-- a small win, from learning speed patterns
our model                1.9 min off
```

**How do we know the differences are real, not luck?** We re-ran the comparison 2,000
times, each time on a reshuffled set of whole days — days, because rides on the same day
share the same weather. If one model wins on almost every reshuffle, the win is real.

## 8. What this means

- **Mostly, ride time = how far you go ÷ about 11.5 km/h.** Arithmetic gets you most of the
  way.
- **Learning adds a real but small polish:** about 3.5 more rides out of every 100 land
  within 2 minutes.
- **The fancier model is no better than the simple one,** so we keep the simple one. It is
  easier to explain, and you can read off exactly what it learned.
- **How you count mistakes matters more than how fancy the model is.**

Compare that with Stage 1: back then, our best model was *worse* than "always say 11
minutes" on the final exam. Now it misses by 1.9 minutes instead of 5.4.

## 9. What is still weak

1. **The weather is daily, not hourly.** A day with rain in the morning counts as rainy at
   5pm, when it may have been sunny. Hourly weather is the last chance for weather to
   matter.
2. **About 3 rides in every 100 have no route history** — brand-new stations or rare
   routes — so they get a rougher guess.
3. **The *average* miss is still about 11.5 minutes,** even though the *typical* miss is 1.9.
   Forgotten bikes drag the average up, and nobody can predict them. The report has to
   explain this, or a reader will think the model is bad.
4. **We only train on 400,000 rides out of 4.9 million.** We never checked whether using
   more would help.
5. **Every run re-reads all 10 million rides from 28 files.** That is most of the
   notebook's 2–3 minutes.
6. **"Statistically clear" is not the same as "matters".** With 2.5 million rides, even a
   3-second difference counts as real. The trees and the simple model differ by about
   that much, and it means nothing in practice.
7. **Round trips follow different rules,** and we handle them with just one yes/no flag.

## 10. What to do next

1. **Write the Stage 2 report.** Lead with the rule we changed (section 3), then use the
   "three guesses" check (section 6) as the main evidence.
2. **Fix the rain sentence.** The Stage 1 report says riders are slower in the rain. The
   data says the opposite.
3. **Report the final exam with its error bars,** and say plainly that the fancy model and
   the simple one tie.
4. **Optional:** try hourly weather.
5. **Optional:** save the cleaned data once, so every rerun skips the slow loading step.
