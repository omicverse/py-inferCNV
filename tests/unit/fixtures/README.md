# Synthetic hspike numerical references

`hspike_rng.tsv` and `hspike_spline_*.tsv` contain synthetic values only.
They were generated with R 4.5.3 and stats 4.5.3. Regenerate from this
directory with:

```sh
Rscript --vanilla generate_hspike_numerics.R
```

The generator fixes Mersenne-Twister / Inversion / Rejection and exercises
mixed RNG consumption, singleton sampling, repeated spline inputs, GCV
and extrapolation. `test_hspike_r_compat.py` reads the saved references;
ordinary Python tests do not require R.

References: [R RNG conventions](https://stat.ethz.ch/R-manual/R-devel/library/base/html/Random.html),
[smooth.spline](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/smooth.spline.html).
