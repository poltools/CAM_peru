
# Initializing the API
import pandas as pd
import os
from openai import OpenAI, AzureOpenAI
api_key = ""

endpoint = os.getenv("ENDPOINT_URL", "")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", api_key)  

# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-05-01-preview",
)

def generate_response(prompt, model):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a classifier that maps text to predefined categories based on detailed descriptions."},
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
    )
    
    return completion.choices[0].message.content.strip()


df = pd.read_excel('provided_docs/TradyAlt Fase1.xlsx', sheet_name='Completos', header=[0, 1])

df_copy = df.copy()
df_copy.columns = df_copy.columns.droplevel(1)

df_copy = df_copy[["Técnica", "Abierta"]].sample(n=300, random_state=42)


prompt_1 = "The following texts consist of descriptions of how people use certain pseudotherapies. Please perform multi-class classification using the following annotation schema of psychological attitude roots: '<SCHEMA>'. Provide the list as a python list as ```[label1, ..., labelN]```. Don't provide further comments. Only the list. This is the text '<REASONS>', and this is the therapy they talk about '<THERAPY>'."


classification_schema_1 = {
  "labels": [
    {
      "name": "Aproximación Holística",
      "description": "Integralidad del ser humano, cuerpo-mente, enfatizando que un estilo de vida positivo, buena salud mental y alimentación adecuada son esenciales en la salud."
    },
    {
      "name": "Tratamientos Naturales",
      "description": "Preferencia por lo natural y libre de químicos, reconociendo la historia y eficacia de los remedios naturales en la medicina moderna."
    },
    {
      "name": "Insatisfacción con la Medicina Convencional",
      "description": "Desconfianza hacia las compañías farmacéuticas y autoridades por corrupción e intereses económicos, pero reconociendo que la desconfianza puede estar justificada en algunos casos."
    },
    {
      "name": "Beneficios y Seguridad de CAM",
      "description": "Percepción de mayor beneficio y seguridad de la medicina alternativa comparada con los tratamientos médicos convencionales."
    },
    {
      "name": "Testimonios Personales",
      "description": "Confianza en experiencias personales como fuente de información, destacando la importancia de considerar las preferencias y experiencias de los pacientes en la práctica médica."
    },
    {
      "name": "Intuición y Preferencias Personales",
      "description": "Valoración de la intuición personal y el conocimiento propio del cuerpo, discutiendo las decisiones médicas teniendo en cuenta las preferencias e intuiciones de los pacientes."
    },
    {
      "name": "Tradicionalismo",
      "description": "Valoración de remedios y prácticas tradicionales como parte integral de la identidad cultural de los individuos, respetando estas tradiciones en la atención médica."
    },
    {
      "name": "Espiritualidad",
      "description": "Respeto a las convicciones religiosas y espirituales en la toma de decisiones médicas, considerando cómo estas creencias influyen en la aceptación de los tratamientos."
    }
  ]
}



prompt_2 = """
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
Razones que apelan a experiencias personales positivas, tanto de la propia persona como de sus familiares, amigos o allegados. Muchas veces se asume que los testimonios son superiores a la evidencia científica porque son “evidencia real” o “de primera mano” en favor de la medicina alternativa. A veces se apela, de forma general, a la buena fama de la medicina alternativa, a su presencia en medios de comunicación, videos, películas, documentales, lecturas o redes sociales, o a que determinada técnica sea mundialmente conocida. 

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


if __name__ == "__main__":
    
  results = []

  extracted_reasons = []
  with open('data/classified_reasons_2203_n=300_rs=42.csv', 'a') as f:
      for index, row in df_copy.iterrows():
          therapy = row["Técnica"]
          reasons = row["Abierta"]
          # prompt_ prompt_1.replace("<THERAPY>", therapy).replace("<SCHEMA>", str(classification_schema))
          prompt_ = prompt_2.replace("<REASONS>", reasons)
          response = generate_response(prompt_, 'gpt-4o')
          print("---")
          print(reasons)
          print(response)
          print("---")
          f.write(f"{row.name}; {therapy}; {reasons}; {response}\n")
  f.close()