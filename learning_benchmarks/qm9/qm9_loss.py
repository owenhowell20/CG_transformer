import torch


def convert_to_ev(x, task):
    # Hartree to eV conversions
    hartree2eV = 27.2114
    unit_conversion = {
        "mu": 1.0,
        "alpha": 1.0,
        "homo": hartree2eV,
        "lumo": hartree2eV,
        "gap": hartree2eV,
        "r2": 1.0,
        "zpve": hartree2eV,
        "u0": hartree2eV,
        "u298": hartree2eV,
        "h298": hartree2eV,
        "g298": hartree2eV,
        "cv": 1.0,
    }
    return unit_conversion[task] * x


def norm2units(x, task):
    # Hartree to meV conversions
    hartree2meV = 27.2114 * 1000
    unit_conversion = {
        "mu": 1.0,
        "alpha": 1.0,
        "homo": hartree2meV,
        "lumo": hartree2meV,
        "gap": hartree2meV,
        "r2": 1.0,
        "zpve": hartree2meV,
        "u0": hartree2meV,
        "u298": hartree2meV,
        "h298": hartree2meV,
        "g298": hartree2meV,
        "cv": 1.0,
    }
    return x * unit_conversion[task]


def mock_task_loss(pred, target, mean, std, FLAGS, use_mean=True):
    l1_loss = torch.sum(torch.abs(pred - target))
    l2_loss = torch.sum((pred - target) ** 2)
    if use_mean:
        l1_loss /= pred.shape[0]
        l2_loss /= pred.shape[0]

    # Rescale the l1 loss to original units
    rescale_loss = norm2units(l1_loss, task=FLAGS.task, mean=mean, std=std)

    return l1_loss, l2_loss, rescale_loss
