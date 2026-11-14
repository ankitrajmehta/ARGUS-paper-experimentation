# Pre-registered judging criteria

Written **before** any image in the review set was opened. Committed before the
first judgement, so the commit timestamp is checkable against the judgement file.

The question being judged is **not** "what did the driver do". It is:

> From the front camera alone, is the correct action for the ego vehicle
> *right now* clearly **a) proceed** or clearly **b) brake**?

## Verdicts

* **`a)`** — proceed is clearly correct
* **`b)`** — brake is clearly correct
* **`unclear`** — the image does not settle it

## What makes `b)` (brake) clear

At least one of:

1. A **red or amber traffic light** governing the ego lane is visible.
2. A **vehicle, pedestrian or cyclist** is in the ego lane close enough that
   continuing would close the gap dangerously.
3. A **stop line or crossing with people on it** directly ahead.
4. The ego vehicle is plainly **in a queue** of stopped traffic.

## What makes `a)` (proceed) clear

All of:

1. **No** red/amber light governing the ego lane.
2. The ego lane ahead is **clear** for a reasonable distance.
3. No pedestrian or cyclist entering or about to enter the ego path.

## What forces `unclear`

Any of these, regardless of the above:

* The governing traffic light is **not visible or not legible**.
* The ego lane ahead is **occluded** so its state cannot be established.
* The scene is a **junction with no visible control** and cross traffic whose
  right of way cannot be determined.
* **Night, heavy rain or glare** sufficient to hide the lane ahead.
* The ego vehicle appears to be **mid-manoeuvre** (turning, lane changing) so
  "proceed vs brake" is not the live question.

## Rules binding the judge

* Judge from **CAM_FRONT only**, the view the decision depends on.
* The DriveLM label is in a **separate file** and stays unopened until every
  judgement is recorded.
* Record the **visual evidence** for each verdict in one short phrase, so a
  human can check the reasoning rather than only the verdict.
* **`unclear` is a valid and expected answer.** Any pressure to resolve a frame
  one way defeats the purpose. Expect a substantial `unclear` share.

## Known limitation, stated up front

The judge is a vision-language model, and the experiment studies
vision-language models. Agreement between this judge and DriveLM does **not**
establish ground truth. It establishes that two independent sources concur, and
the resulting set must be described as *"frames where DriveLM's label and an
independent visual judgement agree"* — never as *"correct labels"*.

Mitigation: a human spot-check of a sample of these judgements. Until that is
done, every number derived from this set carries the caveat above.
