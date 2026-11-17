
"""Exceptions personnalisées du projet.
 
Hiérarchie :
 
    ProjetError
    ├── DonneeInvalideError                  (problèmes liés aux données)
    │   ├── FichierDonneesIntrouvableError
    │   └── ColonneManquanteError
    └── ModeleError                          (problèmes liés au modèle)
        ├── ModeleIntrouvableError
        └── EntrainementImpossibleError
 
Un seul « except ProjetError » attrape toutes les erreurs du projet ;
un « except DonneeInvalideError » attrape seulement celles des données.
"""

class ProjetError(Exception): pass


class DonneeInvalideError(ProjetError): pass

class FichierDonneesIntrouvableError(DonneeInvalideError): pass
class ColonneManquanteError(DonneeInvalideError): pass

class ModeleError(ProjetError): pass

class ModeleIntrouvableError(ModeleError): pass
class EntrainementImpossibleError(ModeleError): pass

