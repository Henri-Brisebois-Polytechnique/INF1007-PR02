
"""Exceptions personnalisées du projet."""

class ProjetError(Exception): pass


class DonneeInvalideError(ProjetError): pass

class FichierDonneesIntrouvableError(DonneeInvalideError): pass
class ColonneManquanteError(DonneeInvalideError): pass

class ModeleError(ProjetError): pass

class ModeleIntrouvableError(ModeleError): pass
class EntrainementImpossibleError(ModeleError): pass

