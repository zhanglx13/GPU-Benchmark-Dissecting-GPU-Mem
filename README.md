# Redo the experiment of Dissecting GPU Memory Hierarchy through Microbenchmarking

Original website can he found [here](http://www.comp.hkbu.edu.hk/~chxw/gpu_benchmark.html).

## Texture Cache

`fine_grain_texture_L1.cu` contains code for all GPU architectures.
Results for Kepler (cc3.5, TITAN) GPU is saved in `result_Kepler_texture_3073_1.txt`.
Results for Pascal (cc6.1, GTX1080) GPU is saved in `result_Pascal_texture_6145_1.txt`.
The texture cache structure can be inferred as

| GPU | cache size | line size | \#Sets | Policy |
| --- | -----------| --------- | ------ | ------ |
| Kepler | 12 KB | 32 B | 4 | LRU |
| Pascal | 24 KB | 32 B | 4 | LRU |
