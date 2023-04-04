## Validate solutions

https://mh2223.danimolina.net/testsol.html

## TODO

| Task                                                             | status          |
| ---------------------------------------------------------------- | --------------- |
| Organize datasets in a more ergonomic way (classes or some sort) | **OK**          |
| Implement 1-NN Classifier                                        | **OK**          |
| Measure performance of classifier 1NN                            | **OK**          |
| Implement _Greedy_ algorithm                                     | **OK** for now  |
| Implement validation                                             | **OK** for now  |
| Implement LS algorithm                                           | Needs Test...   |
| Validity check tests                                             | Not implemented |

## IDEAS

- `fwl.py:78` Maybe do friends and enemies precalculation a bit faster

## Notes

- There's a possibility that a neighbour is exactly the same as the example being evaluated, in that case we will ignore this.

# References

- Precalculate pair-wise distances of examples: https://sparrow.dev/pairwise-distance-in-numpy/
