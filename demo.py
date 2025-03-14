from nvfuser import fusion

f = fusion.Fusion()
fg = fusion.FusionGuard(f)

tv0 = fusion.TensorViewBuilder().n_dims(1).shape([10]).contiguity(True).build()
tv1 = fusion.TensorViewBuilder().n_dims(1).shape([10]).contiguity(True).build()
f.add_input(tv0)
f.add_input(tv1)

tv2 = fusion.ops.add(tv0, tv1)
f.add_output(tv2)

f.print_math()
