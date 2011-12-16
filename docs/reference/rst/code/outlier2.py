import Orange

data = Orange.data.Table("bridges")
outlier_det = Orange.preprocess.outliers.OutlierDetection()
outlier_det.set_examples(data, Orange.distance.instances.EuclideanConstructor(data))
outlier_det.set_knn(3)
z_values = outlier_det.z_values()
for ex,zv in sorted(zip(data, z_values), key=lambda x: x[1])[-5:]:
    print ex, "Z-score: %5.3f" % zv
