# Merge then split vs split then merge

If I have a tensor `T0[I1, I2]`, are the following two schedules equivalent?

- **Schedule 1:** `[I1, I2] -- split -> [I1, I2/4, 4] -- merge -> [I1*(I2/4), 4]`.
- **Schedule 2:** `[I1, I2] -- merge -> [I1*I2] -- split -> [(I1*I2)/4, 4]`.

Where the divisions above are all ceildiv. The answer is no.
Why? Let's see an example where `I1` has extent `2`, and `I2` has extent `5`.

For schedule 1, after schedule, the extents of the leaf domain will be `[2*2, 4]`.
So for this schedule, we will be iterating the tensor as:

```python
T[0, 0], T[0, 1], T[0, 2] , T[0, 3]
T[0, 4], T[0, 5], T[0, 6] , T[0, 7]
T[1, 0], T[1, 1], T[1, 2] , T[1, 3]
T[1, 4], T[1, 5], T[1, 6] , T[1, 7]
```

For schedule 2, after schedule, the extents of the leaf domain will be `[3, 4]`.
So for this schedule, we will be iterating the tensor as:

```python
T[0, 0], T[0, 1], T[0, 2] , T[0, 3]
T[0, 4], T[1, 0], T[1, 1] , T[1, 2]
T[1, 3], T[1, 4], T[2, 0] , T[2, 1]
```

They are clearly not equivalent.

If you are using TMA to access the tensor and wants the out of bound items in smem
being padded as zero automatically by TMA, then schedule 1 should be considered a
2D TMA, and schedule 2 should be considered as 1D TMA.
