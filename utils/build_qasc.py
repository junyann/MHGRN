"""
Script to convert the retrieved HITS into an entailment dataset
USAGE:
 python convert_qasc.py input_file output_file

JSONL format of files
1. input_file -- .data/qasc/{split}.jsonl
{
    "id": "3NGI5ARFTT4HNGVWXAMLNBMFA0U1PG", 
    "question": {
        "stem": "Climate is generally described in terms of what?", 
        "choices": [
            {"text": "sand", "label": "A"}, 
            {"text": "occurs over a wide range", "label": "B"},
            {"text": "forests", "label": "C"}, 
            {"text": "Global warming", "label": "D"}, 
            {"text": "rapid changes occur", "label": "E"}, 
            {"text": "local weather conditions", "label": "F"}, 
            {"text": "measure of motion", "label": "G"}, 
            {"text": "city life", "label": "H"}
        ]
    }, 
    "answerKey": "F", 
    "fact1": "Climate is generally described in terms of temperature and moisture.", 
    "fact2": "Fire behavior is driven by local weather conditions such as winds, temperature and moisture.", 
    "combinedfact": "Climate is generally described in terms of local weather conditions", 
    "formatted_question": "Climate is generally described in terms of what? (A) sand (B) occurs over a wide range (C) forests (D) Global warming (E) rapid changes occur (F) local weather conditions (G) measure of motion (H) city life"
}


2. input_file -- ./data/qasc/{split}_2step.jsonl
{
    "id": "3NGI5ARFTT4HNGVWXAMLNBMFA0U1PG", 

    "choices": [
        {
            "text": "sand", 
            "label": "A", 
            "2-step": [
                 ["Climate Wyoming's climate is generally cool and dry.", "Sand boxes are cool."], 
                 ["Climate A generally moderate climate prevails.", "Sand boils and sand fissures sometimes occur during moderate to large earthquakes."], 
                 ["Climate Omaha's climate can best be described as varied.", "Omaha street crews get their sand trucks filled Monday morning."], 
                 ["Climate Omaha's climate can best be described as varied.", "Sand works best."], 
                 ["Climate Omaha's climate can best be described as varied.", "Coarse builder s sand is the best type of sand to use in a growing mix."], 
                 ["Britain's climate is described as 'equable'.", "Sand eels Another eel typically found around Britain is the sand eel."], 
                 ["Climate Wyoming's climate is generally cool and dry.", "Silence Like Cool Sand Silence Like Cool Sand by Pat Mora First lie in it."], 
                 ["Climate Wyoming's climate is generally cool and dry.", "Sandpipers forage the cool wet sand."], 
                 ["Climate Wyoming's climate is generally cool and dry.", "C . is greatest for dry sand."], 
                 ["Climate A generally moderate climate prevails.", "Simply described, a sand tray is a small sand box designed for indoor use."]
            ]
        }, 
             
        {
            "text": "occurs over a wide range", 
            "label": "B", 
            "2-step": [
                ["Climate is generally described in terms of temperature and moisture.", "Infection occurs over a wide range of soil moisture levels."], 
                ["Electromagnetic radiation occurs in a wide range of wavelengths.", "Important Terms ether electromagnetic wave electromagnetic spectrum electromagnetic radiation 2."], 
                ["Climate is generally described in terms of temperature and moisture.", "Temperature compensation ensures accurate readings over a wide temperature range."], 
                ["Climate is generally described in terms of temperature and moisture.", "C. equisetifolia tolerates a wide range of moisture availability."], 
                ["Climate A wide range of climatic conditions are present in the large geographical range of redbud.", "Cultural or climatic terms derived from geographical proper names are generally lowercased."], 
                ["Climate is generally described in terms of temperature and moisture.", "Crop can be grown at wide temperature range."], 
                ["Eosinophilia occurs in a wide range of conditions.", "Climate is generally described in terms of temperature and moisture."], 
                ["Picture Clarity Clarity is generally described in terms of resolution.", "Range and clarity are both outstanding."], 
                ["Electromagnetic radiation occurs in a wide range of wavelengths.", "Climate is generally described in terms of temperature and moisture."], 
                ["Infection occurs over a wide range of soil conditions.", "Climate is generally described in terms of temperature and moisture."]
            ]
        }, 

        {
            "text": "forests", 
            "label": "C", 
            "2-step": [
                ["Climate is generally described in terms of temperature and moisture.", "Bloom in coniferous forests is geared less to sunlight patterns than to temperature and moisture."], 
                ["Climate is generally described in terms of temperature and moisture.", "Clearcutting increases soil moisture in Engelmann spruce forests."], 
                ["Climate is generally described in terms of temperature and moisture.", "Forests have low albedo, trap heat and moisture."], 
                ["Climate Forests stabilize climate.", "Picture Clarity Clarity is generally described in terms of resolution."], 
                ["Climate Forests stabilize climate.", "Climate is generally described in terms of temperature and moisture."], 
                ["Tropical forests stabilize the world's climate.", "S ub-tropical to tropical, describes the climate generally."], 
                ["Climate Forests stabilize climate.", "Size of smaller properties are generally described in terms of square meters."], 
                ["Forests play a dual role in climate change.", "Role play using Internet terms."], 
                ["Climate Forests stabilize climate.", "Laser surgery generally is used to stabilize vision."], 
                ["People who live in tropical forests, where the climate is very humid, are generally very small.", "People can also be described in terms of where they live."]
            ]
        }, 
        {
            "text": "Global warming", 
            "label": "D", 
            "2-step": [
                ["Climate models predict a slightly drier climate in tropical areas under global warming.", "Climate models generally predict longer, drier summers in interior Alaska."], 
                ["Climate is generally described in terms of temperature and moisture.", "global warming is when worldwide temperature increases"], 
                ["Climate is generally described in terms of temperature and moisture.", "Believers in global warming think that the increased temperature proves that global warming exists."], 
                ["For more on global warming and climate change, see The Pew Center on Global Climate Change .", "All terms described below are subject to change."], 
                ["Climate is generally described in terms of temperature and moisture.", "Global warming is the change of the Earth's temperature."], 
                ["Climate is generally described in terms of temperature and moisture.", "Global Warming Global warming is no game."], 
                ["For more on global warming and climate change, see The Pew Center on Global Climate Change .", "All of the terms described below are subject to change."], 
                ["Global Warming Fact Sheet - Climate changes due to human activities is known as Global Warming.", "Several activities are described in the fact sheet on Demand-Side Management in Hawaii ."], 
                ["For more on global warming and climate change, see The Pew Center on Global Climate Change .", "Terms change constantly."], 
                ["Global warming and climate change result from the greenhouse effect.", "Temperatures on Venus have been described as the result of a run-away greenhouse effect."]
            ]
        }, 
        {
            "text": "rapid changes occur", 
            "label": "E", 
            "2-step": [
                ["Climate is generally described in terms of temperature and moisture.", "Rapid temperature changes and spontaneous storms do occur."], 
                ["Climate is generally described in terms of temperature and moisture.", "Rapid mood changes often occur."], 
                ["Climate is generally described in terms of temperature and moisture.", "Mesic temperature and ustic moisture regimes occur."], 
                ["Climate is generally described in terms of temperature and moisture.", "Rapid physical changes occur during adolescence."], 
                ["Adolescence Moving into adolescence, rapid changes occur.", "Adolescence is generally a period of good health."], 
                ["Adolescence Moving into adolescence, rapid changes occur.", "After adolescence the months are generally considered negligible."], 
                ["Adolescence Moving into adolescence, rapid changes occur.", "Climate is generally described in terms of temperature and moisture."], 
                ["Rapid and extreme changes can occur fairly often.", "Climate is generally described in terms of temperature and moisture."], 
                ["Rapid changes in the river level can occur without warning.", "Climate is generally described in terms of temperature and moisture."], 
                ["Rapid mood changes often occur.", "Heat is often described in cgs terms."]
            ]
        }, 
        {
            "text": "local weather conditions", 
            "label": "F", 
            "2-step": [
                ["Climate is generally described in terms of temperature and moisture.", "Fire behavior is driven by local weather conditions such as winds, temperature and moisture."], 
                ["Climate is generally described in terms of temperature and moisture.", "Temperature and moisture conditions also affect sensitivity."], 
                ["Climate is generally described in terms of temperature and moisture.", "Local conditions of air temperature and relative humidity dictate the final moisture level."], 
                ["Climate is generally described in terms of temperature and moisture.", "Weather conditions - temperature, humidity, cloud cover."], 
                ["Local weather conditions often vary.", "Climate is generally described in terms of temperature and moisture."], 
                ["Nepal Climate Nepal's weather is generally predictable and pleasant.", "Explore Nepal - local English newspaper published fortnightly."], 
                ["All readings reflect local weather conditions.", "Climate is generally described in terms of temperature and moisture."], 
                ["Introduction to Climate Introduction Climate is the average weather conditions.", "CD-ROM described on the Introduction page."], 
                ["Introduction to Climate Introduction Climate is the average weather conditions.", "With the introduction of a new concept comes the introduction of new terms."], 
                ["Introduction to Climate Introduction Climate is the average weather conditions.", "Introduction Hindus generally cremate their dead."]
            ]
        }, 
        {
            "text": "measure of motion", 
            "label": "G", 
            "2-step": [
                ["Picture Clarity Clarity is generally described in terms of resolution.", "Clarity Clarity Clarity is a measure of a diamond's purity."], 
                ["Climate is generally described in terms of temperature and moisture.", "Temperature is a measure of molecular motion."], 
                ["Climate is generally described in terms of temperature and moisture.", "Temperature is a measure of average molecular motion."], 
                ["Climate is generally described in terms of temperature and moisture.", "Temperature is a measure of the motion of molecules."], 
                ["Climate is generally described in terms of temperature and moisture.", "Temperature is the measure of relative motion of molecules."], 
                ["Picture Clarity Clarity is generally described in terms of resolution.", "Uniqueness provides a measure of clarity."], 
                ["Robotic motion is described in terms of degrees of freedom.", "Engineers measure dexterity by degrees of freedom."], 
                ["Speed is the measure of motion.", "Climate is generally described in terms of temperature and moisture."], 
                ["Size of smaller properties are generally described in terms of square meters.", "Standard guest rooms measure 26.4 square meters and offer a queen-size bed."], 
                ["Robotic motion is described in terms of degrees of freedom.", "Characters are generally robotic."]
            ]
        }, 
        {
            "text": "city life", 
            "label": "H", 
            "2-step": [
                ["Climate is generally described in terms of temperature and moisture.", "Temperature and moisture are the primary factors controlling the flea life cycle."], 
                ["Climate is generally described in terms of temperature and moisture.", "Room temperature shelf life of bread products is extended, while retaining moisture and consistency."], 
                ["Climate Omaha's climate can best be described as varied.", "Teamwork - to bring communication to City Hall and a common goal of what s best for Omaha."], 
                ["Climate Omaha's climate can best be described as varied.", "Nebraska's largest city is Omaha."], 
                ["Climate Omaha's climate can best be described as varied.", "Another major city is Omaha."], 
                ["Instructions Climate South Africa's climate is generally sunny and pleasant.", "For more information about South Africa see City."], 
                ["Quality of Life Respondents generally speak highly about the city.", "Climate is generally described in terms of temperature and moisture."], 
                ["Climate A generally moderate climate prevails.", "Sometimes, life can best be described in terms of velocity."], 
                ["Climate A generally moderate climate prevails.", "Rules fail, life experience prevails."], 
                ["Climate Omaha's climate can best be described as varied.", "City of Omaha - The city's official website has the facts on Omaha."]
            ]
        }
    ]
}

3. output_file -- ./data/qasc/statement/{split}.statement.jsonl
{
    "id": "3NGI5ARFTT4HNGVWXAMLNBMFA0U1PG", 
    "question": {
        "stem": "Climate is generally described in terms of what?", 
        "choices": [
            {"text": "sand", "label": "A"}, 
            {"text": "occurs over a wide range", "label": "B"},
            {"text": "forests", "label": "C"}, 
            {"text": "Global warming", "label": "D"}, 
            {"text": "rapid changes occur", "label": "E"}, 
            {"text": "local weather conditions", "label": "F"}, 
            {"text": "measure of motion", "label": "G"}, 
            {"text": "city life", "label": "H"}
        ]
    },
    "answerKey": "F", 
    "statements":[
        {label:false, stem: "Climate is generally described in terms of sand."},
        {label:false, stem: "Climate is generally described in terms of occurs over a wide range."},
        {label:false, stem: "Climate is generally described in terms of forests."}
        {label:false, stem: "Climate is generally described in terms of Global warming."}
        {label:false, stem: "Climate is generally described in terms of forests."}
        {label:false, stem: "Climate is generally described in terms of rapid changes occur."}
        {label:true, stem: "Climate is generally described in terms of local weather conditions."}
        {label:false, stem: "Climate is generally described in terms of measure of motion."}
        {label:false, stem: "Climate is generally described in terms of city life."}
    ]
}
"""

import json
import re
import sys
import itertools
from tqdm import tqdm

__all__ = ['convert_to_entailment']

# String used to indicate a blank
BLANK_STR = "___"


def build_qasc(qa_file: str, qa_2step_file: str, output_file: str):

    qa_2step_dict = {}
    nrow_qa_2step = sum(1 for _ in open(qa_2step_file, 'r'))
    with open(qa_2step_file, 'r') as qa_2step_handle:
        for qa_2step_line in tqdm(qa_2step_handle, total=nrow_qa_2step, desc='Loading qa_2step lines'):
            qa_2step_json_line = json.loads(qa_2step_line)
            qa_2step_dict[qa_2step_json_line['id']] = qa_2step_json_line

    nrow = sum(1 for _ in open(qa_file, 'r'))
    with open(output_file, 'w') as output_handle, open(qa_file, 'r') as qa_handle:
        for line in tqdm(qa_handle, total=nrow, desc='Building QA 2-step dataset'):
            json_line = json.loads(line)
            output_dict = build_qajson(json_line, qa_2step_dict[json_line['id']])
            output_handle.write(json.dumps(output_dict))
            output_handle.write("\n")
    print(f'QASC 2-step dataset saved to {output_file}\n')

# Convert the QA file json to output dictionary containing premise and hypothesis
def build_qajson(qa_json: dict, qa_2step_json: dict):
    choices = qa_json["question"]["choices"]
    for i, choice in enumerate(choices):
        evidence_sentences = list(itertools.chain.from_iterable(qa_2step_json['choices'][i]['2-step']))
        evidence = []
        [evidence.append(x) for x in evidence_sentences if x not in evidence]
        evidence = ' '.join(evidence)
        create_output_dict(qa_json, evidence, i)
    qa_json['question']['stem'] = ''
    return qa_json

# Create the output json dictionary
def create_output_dict(input_json: dict, evidence: str, choice_idx: int) -> dict:
    question = input_json['question']['stem']
    choice = input_json['question']['choices'][choice_idx]
    answer = choice['text']
    choice['text'] = 'Q: {} A: {} </s> {}'.format(question, answer, evidence)
    return input_json

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Provide at least two arguments: "
                         "json file with hits, output file name")
    build_qasc(sys.argv[1], sys.argv[2], sys.argv[3])