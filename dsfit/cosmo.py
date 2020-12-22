def get_cosmo(name):
    import colossus.cosmology.cosmology
    return colossus.cosmology.cosmology.Cosmology(
        name=name,
        **colossus.cosmology.cosmology.cosmologies[name]
    )
