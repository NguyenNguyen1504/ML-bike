# The whole project, in plain words

No jargon. If a word has to be technical, it is explained right there.

---

## 1. What are we even trying to do?

You are standing at a city bike rack. Before you ride, you want to know:
**how long will this ride take?**

We try to guess that number before the ride happens, using only three easy things:

- what day it is (Monday? Saturday?)
- what time it is (8 in the morning? 5 in the evening?)
- what the weather is like (warm? raining?)

That is the whole idea. Guess the ride length from the day, the time, and the sky.

## 2. What do we have to work with?

- **Every city bike ride** in Helsinki and Espoo for four summers (April–October, 2022
  to 2025). About **10 million rides**. Each one tells us when it started and how long
  it took.
- **The weather for every one of those days** — temperature and rain — from the Finnish
  weather service.

We glue the two together by date. Now every ride knows what the weather was that day.

## 3. One rule we gave ourselves

We do **not** tell the model where the rider is going.

Why? Because that would be cheating. If you know the start and the end point, you
basically know the distance, and distance almost tells you the time all by itself. We
want to find out what the day, time and weather can do **on their own**.

Remember this rule. It explains almost everything that happens later.

## 4. What the notebooks actually do, step by step

1. **Read** all 28 monthly files — 10.1 million rides.
2. **Throw away the broken ones.** Rides under 10 seconds or 10 metres are someone
   pulling a bike out and pushing it straight back. About 9.8 million rides survive —
   97 out of every 100.
3. **Look at the data.** A normal ride is about **11 minutes**.
4. **Look for patterns.** Rides are a bit longer in summer, a bit longer at weekends,
   a bit shorter when it rains.
5. **Split by year, like exams.** Learn from 2022 and 2023. Check ourselves on 2024.
   Keep 2025 locked in a drawer as the final exam, never peeked at until the very end.
6. **Teach several models** and see which guesses best.
7. **Open the drawer** and take the final exam once.

There are two notebooks, and they overlap:

- `ML_bike.ipynb` — the full story, all 11 steps.
- `ML_bike_stage1/ML_bike_stage1.ipynb` — a shortened copy, steps 1–7 only, which became
  the appendix of the Stage 1 report.

## 5. The one big problem: forgotten bikes

Most rides are about 11 minutes. But some "rides" last **days**. The longest one in our
data lasted **194 days**. Nobody cycled for 194 days. Somebody forgot to put the bike
back.

Here is why that matters so much.

When you teach a model, you tell it how to count its mistakes. The usual way is called
**squared error**: being wrong by 2 counts as 4, being wrong by 10 counts as 100. Big
mistakes are punished enormously.

Now put a 194-day ride in front of it. The model would rather twist its whole answer than
be hugely wrong about that one bike. And the numbers prove it:

> **0.3% of the rides cause 99.7% of the pain the model is trying to reduce.**

So the model is not really learning about cycling. It is learning about forgotten bikes.

Our fix: **count mistakes more kindly**. Being wrong by 10 counts as 10, not 100. These
kinder rules are called **Huber** and **absolute error**. The weird rides still exist —
we did not delete them — they just no longer get to shout over everyone else.

## 6. What came out

How far off each guess is, on average, in minutes (smaller is better):

| How we guess | How far off |
|---|---|
| Pick a random real ride and say that | 20.9 min |
| Always say "11 minutes", ignore everything | 14.29 min |
| Our best model (day + time + weather) | 14.21 min |
| A fancy model (boosted trees) | 14.24 min |
| The most flexible guess that is even possible here | 14.25 min |

Read that table slowly, because it is the whole result:

```
random guessing   20.9
                    |  <-- a huge win, just from saying "about 11 minutes"
always say 11     14.29
                    |  <-- a tiny win, from ALL the day/time/weather cleverness
our best model    14.21
```

Going from random to "always say 11 minutes" saves about **6 and a half minutes**.
Going from there to our cleverest model saves about **5 more seconds**.

We also tested whether the fancy model was being held back by not being fancy enough. It
was not. We built the most flexible guesser possible with these three facts — one that
can memorise any pattern at all — and it did no better.

## 7. What this means

**Day, time and weather barely tell you anything about how long one particular ride
takes.**

They do tell you something real about the *typical* ride: summer rides are longer, rainy
rides are shorter, weekend rides are longer. Those patterns are genuinely there. They are
just tiny next to the real question. Two people cycling at the same hour, on the same
day, in the same weather still differ enormously — because one is going three streets and
the other is going across the city.

And we are the ones who decided not to tell the model where anyone is going (step 3).

**This is an honest result, not a failed project.** The useful thing we can say is not
"we built a good predictor" — we did not. It is: *here is exactly how much these features
can possibly be worth, and it is almost nothing.* That is a real finding, and we can prove
it instead of just claiming it.

## 8. What is weak, wrong, or wasteful

Things we should fix or admit out loud:

1. **Our "winner" won by 2 seconds — and the race was unfair.** We called Huber the best
   model. The runner-up lost by 0.03 minutes, and it was under-trained: we only let it
   practise 50 rounds while the winner got 300. We should not claim a winner from that.
2. **On the final exam our model was worse than "always say 11 minutes"** for a typical
   ride (off by 5.66 min versus 5.43). We must say this plainly in the report.
3. **We report two numbers that mean nothing here** (RMSE and R²). Both are built out of
   squared error, so the forgotten bikes swallow them. R² comes out as 0.00 for every
   single model, which makes it look like nothing works, for the wrong reason.
4. **Every run re-reads all 10 million rides** from 28 files. That is about 40 seconds of
   pure waiting before anything happens, and 3.5 minutes for the whole notebook.
5. **We only train on 400,000 rides out of 4.9 million** because some models are slow. We
   never checked whether using more would change anything.
6. **The two notebooks share their first 7 steps by copy-paste.** The moment someone edits
   one, they disagree, and the report's appendix stops matching the real work.
7. **The weather is daily, not hourly.** A day with rain in the morning counts as rainy at
   5pm, when it may have been sunny. This blurs the one feature we would most expect to
   matter.
8. **Our most flexible guesser ran out of data.** Splitting the rides by all five facts at
   once left too few rides in each group — only about half of the checking rides landed in
   a group big enough to trust.

## 9. What should change

In the order worth doing it:

1. **Say the honest result out loud and make it the headline.** "These features are worth
   about 1% of the achievable improvement, and here is the proof" is a stronger report
   than a vague claim of success.
2. **Stop claiming a best model.** Say the kind-loss models are equal, and that we chose
   Huber because it trains reliably. Fix the under-trained one or drop the comparison.
3. **Save the cleaned data once** to a single file, and load that instead of re-reading 28
   files every time. One line of code, and the 40-second wait disappears.
4. **Use hourly weather instead of daily.** This is the one change that could genuinely
   improve the predictions, because rain at 5pm is what actually matters.
5. **Keep RMSE and R² if the course wants them, but write one sentence saying why they are
   meaningless here.** Otherwise a reader sees R² = 0 and assumes we did something wrong.
6. **Make the two notebooks share one source** so the appendix cannot drift away from the
   real analysis.
7. **Decide what to do about the destination.** Our own rule says leave it out — but we
   should at least mention what it would be worth, so the reader knows we know.
