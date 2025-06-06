from train import wrap_main
from flags import get_flags

num_points = [512, 1024, 2048, 4096]  ### standard oom at 2048
model_types = ["dgcnn", "SE3Hyena", "Standard"]
tasks = ["Grasp", "Normal"]

for num_point in num_points:
    for task in tasks:
        for modelname in model_types:
            FLAGS, UNPARSED_ARGV = get_flags()
            FLAGS.resolution = num_point
            FLAGS.model = modelname
            FLAGS.task = task

            wrap_main(FLAGS, UNPARSED_ARGV)
