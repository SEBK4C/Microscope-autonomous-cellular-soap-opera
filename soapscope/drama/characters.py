"""Persistent character identities for tracked microbes.

Every track id gets a stable name, honorific and soap-opera archetype the
moment it appears, and keeps it for life. Deterministic given a seed so runs
are reproducible (the autoresearch loop compares like with like).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

TITLES = ["Contessa", "Baron", "Lady", "Sir", "Dr.", "Madame", "Count",
          "Duke", "Duchess", "Professor", "Reverend", "Captain"]

GIVEN = ["Brad", "Vesper", "Chad", "Ophelia", "Dmitri", "Cordelia", "Blaine",
         "Seraphina", "Rex", "Guadalupe", "Thaddeus", "Isolde", "Basil",
         "Delphine", "Roderick", "Anastasia", "Lance", "Genevieve", "Cyrus",
         "Magnolia", "Sterling", "Ramona", "Percival", "Bianca"]

SURNAMES = ["Paramecium", "von Blobsworth", "Ciliato", "Amoebini", "Flagellum",
            "Diatomsky", "Euglenova", "Rotiferi", "Pseudopod", "Membranski",
            "Vacuole", "Protozoa", "Slimewater", "Micrococcus", "Wigglesworth"]

ARCHETYPES = [
    ("the schemer",     "plotting"),
    ("the ingénue",     "wide-eyed"),
    ("the villain",     "menacing"),
    ("the heartthrob",  "smouldering"),
    ("the diva",        "insufferable"),
    ("the underdog",    "plucky"),
    ("the femme fatale","dangerous"),
    ("the golden child","insufferably perfect"),
    ("the black sheep", "misunderstood"),
    ("the amnesiac",    "confused"),
]


@dataclass
class Character:
    track_id: int
    name: str
    archetype: str
    trait: str
    pronoun: str

    @property
    def short(self) -> str:
        # Last token of the name — "Contessa Vesper Paramecium" -> "Paramecium".
        return self.name.split()[-1]


class CharacterRegistry:
    def __init__(self, seed: int = 7):
        self._rng = random.Random(seed)
        self._by_track: Dict[int, Character] = {}
        self._used_names: set = set()

    def get(self, track_id: int) -> Character:
        c = self._by_track.get(track_id)
        if c is not None:
            return c
        # Draw a fresh, unused name deterministically.
        for _ in range(200):
            name = f"{self._rng.choice(TITLES)} {self._rng.choice(GIVEN)} {self._rng.choice(SURNAMES)}"
            if name not in self._used_names:
                break
        self._used_names.add(name)
        arche, trait = self._rng.choice(ARCHETYPES)
        pronoun = self._rng.choice(["they", "she", "he"])
        c = Character(track_id=track_id, name=name, archetype=arche,
                      trait=trait, pronoun=pronoun)
        self._by_track[track_id] = c
        return c

    def known(self) -> List[Character]:
        return list(self._by_track.values())
