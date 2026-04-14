"""
Prompt templates and classifier runners for the CAM-Peru pipeline.

Three classification passes are defined here:

1. ``REASON_EXTRACTION_PROMPT``
   Condenses a raw open-ended justification into a list of nominalised
   reasons (e.g. ``"desconfío de las vacunas" -> "desconfianza hacia vacunas"``).

2. ``DOCUMENT_LEVEL_PROMPT`` (``prompt_document``)
   Assigns one or more of the 10-category schema labels (see
   :data:`cam_peru.config.CATEGORIES_ES`) to an entire open-ended response.

3. ``MICRO_REASON_PROMPT`` (``prompt_micro``)
   Assigns exactly one most-specific category to an already-extracted
   nominalised reason. Used for the sub-argument / micro-classification
   pass whose outputs are aggregated back to the document level.

All three share the same JSON-schema-style output convention: a Python
list of integer category codes, e.g. ``[1, 5]`` — never free-form text.
"""

from __future__ import annotations

import ast
from typing import Iterable, Optional

from .config import (
    CATEGORIES_ES,
    CLASSIFICATION_MAX_TOKENS,
    CLASSIFICATION_SEED,
    CLASSIFICATION_TEMPERATURE,
    CLASSIFICATION_TOP_P,
)
from .llm_client import chat_completion, get_azure_client, get_deployment_name

# --------------------------------------------------------------------------- #
# Reason extraction — condense free-text into nominalised reasons.            #
# --------------------------------------------------------------------------- #
REASON_EXTRACTION_PROMPT = """The following texts consist of descriptions of how people use certain pseudotherapies.
Condense these texts into a list of reasons why they use these pseudotherapies.
Each reason should be expressed in a few nominalized terms.
For example `I don't trust vaccines`-> `vaccine distrust`.
These reasons should be generic enough so that they can be identfied in other texts as well.
Provide the list as a python list. Don't provide further comments. Only the list. Make sure this list is as complete and condense as possible. Do not include the therapy name in the list. Do not generate excessive reasons.
This is the text '<REASONS>', and this is the therapy they talk about '<THERAPY>'.

 Please do this in its original Spanish language."""


# --------------------------------------------------------------------------- #
# Document-level classification (prompt used for the published results).      #
# --------------------------------------------------------------------------- #
DOCUMENT_LEVEL_PROMPT = """
Clasifica el siguiente texto según la razón principal que expresa a favor de la medicina alternativa. Elige una o más categorías que consideres adecuadas. No intentes ser exhaustivo. Sé estricto en tu selección.

Categorías posibles:


1. Creencias paranormales
Estas razones se caracterizan por referir a entidades o procesos paranormales, espirituales, energéticos o religiosos. Estas razones suelen expresar confianza en personas con poderes especiales como curanderos o chamanes, así como creencia en la canalización y estabilización de la energía vital o en el uso de puntos vitales. También se incluye la creencia en enfermedades sin un fundamento físico plausible como la envidia, el mal, la maldad, el susto, las maldiciones, la vergüenza o el mal de ojo. También la creencia en técnicas diagnósticas sin fundamento físico, como el reflejo de la enfermedad en un cuy o huevo.

2. Desconfianza en la medicina convencional
Estas razones expresan críticas, desconfianza, insatisfacción o ineficacia de la medicina convencional. Se critican los medicamentos, los tratamientos y los diagnósticos médicos por ser ineficaces, imprecisos, o por tener efectos adversos o secundarios negativos. A veces se critica la medicina convencional por ser superficial o impersonal, prefiriéndose el enfoque más profundo, integral u holístico de la medicina alternativa. A veces se expresa desconfianza en los intereses económicos de las empresas farmacéuticas y las autoridades sanitarias.

3. Uso de productos naturales
Estas razones expresan preferencia por el uso de productos, remedios o tratamientos naturales. Se suele apelar a que los productos o alimentos naturales están libres de químicos o promueven la autocuración con las propiedades naturales del propio cuerpo. También se apela a una mayor conexión con la tierra o respeto por la naturaleza. Lo natural es visto como inherentemente positivo y beneficioso.

4. Beneficios personales y efectividad clínica
Estas razones se basan tanto en los beneficios personales y como en la efectividad clínica de la medicina alternativa. Además de la efectividad para el tratamiento de diversas dolencias como dolor, fiebre, desalineación ósea o luxaciones, también se mencionan beneficios personales fuera del contexto clínico como una mayor relajación, placer, bienestar, y la reducción del estrés o de la ansiedad.

5. Testimonios y experiencias personales
Razones que apelan a experiencias personales positivas, tanto de la propia persona como de sus familiares, amigos o allegados. Muchas veces se asume que los testimonios son superiores a la evidencia científica porque son "evidencia real" o "de primera mano" en favor de la medicina alternativa. A veces se apela, de forma general, a la buena fama de la medicina alternativa, a su presencia en medios de comunicación, videos, películas, documentales, lecturas o redes sociales, o a que determinada técnica sea mundialmente conocida.

6. Tradicionalismo
Estas razones expresan preferencia por tratamientos y prácticas tradicionales relacionadas con el cuidado de la salud. Estas técnicas y prácticas tradicionales sueles ser presentadas como superiores o equivalentes a la medicina convencional al ser milenarias, antiguas o por reflejar la sabiduría de los ancestros. Muchas veces son valoradas por ser una tradición familiar pasada de generación en generación. Puede hacerse referencia a varias tradiciones culturales, como la peruana, la china o la asiática.

7. Accesibilidad y comodidad
Estas razones se refieren a factores prácticos de la medicina alternativa como una mayor accesibilidad, un precio asequible o comodidad al agendar una cita. Suelen hacerse referencias a dificultades para acceder a la medicina convencional, como largas colas o tiempos de espera, la lejanía de los consultorios médicos o la necesidad de gestionar seguros médicos. También se hace referencia a que la medicina alternativa permite evitar cirugías o el uso de medicamentos indeseados.

8. Credenciales científicas y profesionales
Estas razones expresan la creencia en que la medicina alternativa está respaldada por estudios científicos confiables, basados en hechos observables reales, muchas veces con referencias a la base científica o anatómica de estas prácticas. Los practicantes son percibidos como prestigiosos, experimentados y respaldados por credenciales profesionales. A veces estas razones defienden la medicina alternativa debido a su reconocimiento dentro de la medicina convencional, incluyendo su presencia en instituciones médicas oficiales o la recomendación o práctica por parte de médicos convencionales.

9. Ausencia de efectos secundarios
Razones por las cuales se prefiere la medicina alternativa debido a su ausencia de efectos secundarios, inocuidad o seguridad. Suele apelarse a que los tratamientos alternativos no son invasivos, de modo que no generan afectos adversos o imprevistos. También suele apelarse a la precisión de los tratamientos y a su efecto directo sobre la zona afectada, de modo que no afecta a otras zonas del cuerpo.

Categoría extra: Razones imposibles de categorizar.

Solapamientos: En caso de creencia en efectividad de una enfermedad paranormal, colocarlo en la categoría de paranormal.

Categoría extra:
10. Razones imposibles de categorizar: Si no es posible ubicar el texto en ninguna de las categorías anteriores.

Regla especial: Si se habla de la efectividad de una enfermedad paranormal, clasifícalo como "Creencias paranormales".

Devuelve el resultado como una lista de Python con los números de las categorías, por ejemplo: [1, 5]. No devuelvas comentarios adicionales, sólo la lista.


Este es el texto a classificar:
<REASONS>
"""


# --------------------------------------------------------------------------- #
# Micro-classification — one category per short nominalised reason.           #
# --------------------------------------------------------------------------- #
MICRO_REASON_PROMPT = """
Clasifica las siguientes razones que las personas dan para preferir la medicina alternativa.

Lee el texto con atención y asigna las categorías que correspondan.

INSTRUCCIONES GENERALES:

- Clasifica siempre en la categoría MÁS ESPECÍFICA y DIRECTA posible.
- NO combines categorías salvo que ambas sean absolutamente necesarias.
- EVITA usar la categoría 10 salvo que realmente ninguna otra sea aplicable.
- Si tienes dudas, revisa las guías y ejemplos al final.

CATEGORÍAS POSIBLES:

1. Creencias paranormales
Creencias espirituales, energéticas, mágicas o sobrenaturales. También si menciona rituales, equilibrio espiritual, vibraciones, fuerzas místicas o canalización de energía.

2. Desconfianza en la medicina convencional
Críticas a doctores, hospitales, fármacos, falta de efectividad, superficialidad, intereses económicos o desconfianza en la ciencia médica.

3. Uso de productos naturales
Preferencia por lo natural, ecológico, puro, limpio o sin químicos. Incluye alusión a remedios caseros, plantas medicinales, tratamientos "naturales".

4. Beneficios personales y efectividad clínica
Resultados o beneficios obtenidos, físicos o emocionales, incluyendo frases generales sobre salud, prevención, tratamiento, bienestar o cuidado corporal.

Ejemplos: "mejorar salud", "tratamiento de enfermedades", "prevenir dolencias", "relajación", "autocuidado", "prevenir problemas", "sanar".

5. Testimonios y experiencias personales
Experiencias propias o de personas conocidas. Uso de "yo", "mi mamá", "mi amigo", "vi que funciona", "me ayudó".

6. Tradicionalismo
Valoración de prácticas antiguas, ancestrales, culturales o transmitidas por generaciones. Incluye referencias a "sabiduría ancestral", "métodos milenarios" o tradiciones de algún país.

7. Accesibilidad y comodidad
Referencias a precio, facilidad, rapidez, ubicuidad, cercanía, disponibilidad, menor espera o evitar burocracia.

8. Credenciales científicas y profesionales
Mención de ciencia, estudios, médicos, universidades, instituciones, validación científica o profesionalismo. Incluye frases como "base científica", "avalado por médicos", "conocimiento especializado".

9. Ausencia de efectos secundarios
Valoración de seguridad, no invasividad, inocuidad, protección del cuerpo, sin efectos adversos.

10. Razones abstractas o no relacionadas con la salud
Solo si:
- No menciona salud, beneficios, ciencia, tradición, naturaleza, espiritualidad, accesibilidad ni experiencias.
- Frases existenciales, filosóficas, emocionales, crecimiento personal, curiosidad, paz, reflexión, etc.

GUÍA PARA FRASES VAGAS O ABSTRACTAS:

Si una razón no menciona directamente una palabra clave, pero sugiere claramente la idea central de una categoría → Clasifícalo en esa categoría.

Solo usa la categoría 10 si, después de pensar razonablemente, ninguna otra categoría aplica.

EJEMPLOS DE FRASES DUDOSAS Y SU CLASIFICACIÓN:

- "trabajo integral del cuerpo" → 4
- "armonización interna" → 4
- "conocer mejor mi cuerpo" → 4
- "sabiduría de generaciones" → 6
- "método aprendido de ancestros" → 6
- "fácil de acceder" → 7
- "mayor profesionalismo" → 8
- "paz interior" → 10
- "reflexión personal" → 10
- "crecimiento espiritual" → 1

SOLAPAMIENTOS:

- Beneficio + experiencia → Categoría 5
- Beneficio general sin experiencia → Categoría 4
- Ciencia o estudios → Categoría 8
- Enfermedad paranormal → Categoría 1
- Salud/técnica corporal implícita → Categoría 4

FORMATO DE RESPUESTA:

Devuelve el resultado como una lista de Python con los números de las categorías, por ejemplo: [1, 5]. No devuelvas comentarios adicionales, solo la lista.

Este es el texto a clasificar. Enmarcalo dentro de una expresión como "Uso terapias alternativas para / por ...":

<REASONS>
"""


# --------------------------------------------------------------------------- #
# Runners                                                                     #
# --------------------------------------------------------------------------- #
def parse_category_list(raw: str) -> list[int]:
    """Parse the model's ``[1, 5]``-style output into a list of ints.

    Returns an empty list if parsing fails — the caller can then decide
    whether to retry, skip, or route the document to the Miscellaneous
    bucket. This is deliberately permissive to avoid crashing on minor
    formatting deviations.
    """
    raw = raw.strip()
    # Some deployments occasionally wrap output in markdown — strip fences.
    if raw.startswith("```"):
        raw = raw.strip("`")
        # strip optional "python" language tag
        if raw.lower().startswith("python"):
            raw = raw[len("python"):]
        raw = raw.strip()
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    out: list[int] = []
    for x in parsed:
        try:
            code = int(x)
        except (TypeError, ValueError):
            continue
        if code in CATEGORIES_ES:
            out.append(code)
    return out


def classify_reason(
    reason: str,
    therapy: Optional[str] = None,
    client=None,
    prompt_template: str = DOCUMENT_LEVEL_PROMPT,
) -> list[int]:
    """Classify a single reason string into one or more category codes.

    Parameters
    ----------
    reason
        Either a raw open-ended response (for :data:`DOCUMENT_LEVEL_PROMPT`)
        or a short nominalised phrase (for :data:`MICRO_REASON_PROMPT`).
    therapy
        Optional technique name, only consumed by templates that include
        ``<THERAPY>``.
    client
        An :class:`openai.AzureOpenAI` instance; auto-constructed via
        :func:`cam_peru.llm_client.get_azure_client` if not supplied.
    prompt_template
        One of the ``_PROMPT`` constants in this module.
    """
    client = client or get_azure_client()
    prompt = prompt_template.replace("<REASONS>", reason)
    if therapy is not None:
        prompt = prompt.replace("<THERAPY>", therapy)

    raw = chat_completion(
        client,
        prompt,
        model=get_deployment_name(),
        temperature=CLASSIFICATION_TEMPERATURE,
        top_p=CLASSIFICATION_TOP_P,
        max_tokens=CLASSIFICATION_MAX_TOKENS,
        seed=CLASSIFICATION_SEED,
    )
    return parse_category_list(raw)


def extract_reasons(
    text: str,
    therapy: str,
    client=None,
) -> str:
    """Condense a free-text justification into a list of nominalised reasons.

    Returns the raw model output (a Python-list-formatted string) — callers
    typically persist this directly and parse later with
    :func:`process_text.process_annotations` or :func:`ast.literal_eval`.
    """
    client = client or get_azure_client()
    prompt = REASON_EXTRACTION_PROMPT.replace("<REASONS>", text).replace(
        "<THERAPY>", therapy
    )
    return chat_completion(
        client,
        prompt,
        temperature=CLASSIFICATION_TEMPERATURE,
        top_p=CLASSIFICATION_TOP_P,
    )


__all__ = [
    "REASON_EXTRACTION_PROMPT",
    "DOCUMENT_LEVEL_PROMPT",
    "MICRO_REASON_PROMPT",
    "parse_category_list",
    "classify_reason",
    "extract_reasons",
]
