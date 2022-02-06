# utilities for processing scripts

def _overwrite_dataset_if_exists(group, name, data, verbose=True):
    """Creates a new dataset, overwriting any previous
    dataset with the same name. Caution - destructive.
    In future we may want to have an option in the config
    to prevent this."""

    if name in group:
        if verbose: print(f"Overwriting dataset {group}/{name}")
        del group[name]
    group.create_dataset(name=name, data=data)

def _overwrite_group_if_exists(supergroup, group, verbose=True):
    """Creates a new group, overwriting any previous
    group with the same name. Caution - destructive.
    In future we may want to have an option in the config
    to prevent this."""

    if group in supergroup:
        if verbose: print(f"Overwriting group {supergroup}/{group}")
        del supergroup[group]
    newgroup = supergroup.create_group(group)
    return newgroup