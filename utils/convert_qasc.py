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
"""

import json
import re
import sys
import itertools
from tqdm import tqdm

__all__ = ['convert_to_entailment']

# String used to indicate a blank
BLANK_STR = "___"


def convert_to_entailment(qa_file: str, output_file: str, ans_pos: bool=False):
    print(f'converting {qa_file} to entailment dataset...')
    nrow = sum(1 for _ in open(qa_file, 'r'))
    with open(output_file, 'w') as output_handle, open(qa_file, 'r') as qa_handle:
        for line in tqdm(qa_handle, total=nrow, desc='Processing qa lines'):
            json_line = json.loads(line)
            output_dict = convert_qajson_to_entailment(json_line, ans_pos)
            output_handle.write(json.dumps(output_dict))
            output_handle.write("\n")
    print(f'converted statements saved to {output_file}\n')


# Convert the QA file json to output dictionary containing premise and hypothesis
def convert_qajson_to_entailment(qa_json: dict, ans_pos: bool):
    question_text = qa_json["question"]["stem"]
    choices = qa_json["question"]["choices"]
    for i, choice in enumerate(choices):
        choice_text = choice["text"]
        pos = None
        if not ans_pos:
            statement = create_hypothesis(get_fitb_from_question(question_text), choice_text, ans_pos)
        else:
            statement, pos = create_hypothesis(get_fitb_from_question(question_text), choice_text, ans_pos)
        create_output_dict(qa_json, statement,  choice["label"] == qa_json.get("answerKey", "A"), ans_pos, pos)

    return qa_json


# Get a Fill-In-The-Blank (FITB) statement from the question text. E.g. "George wants to warm his
# hands quickly by rubbing them. Which skin surface will produce the most heat?" ->
# "George wants to warm his hands quickly by rubbing them. ___ skin surface will produce the most
# heat?
def get_fitb_from_question(question_text: str) -> str:
    fitb = replace_wh_word_with_blank(question_text)
    if not re.match(".*_+.*", fitb):
        # print("Can't create hypothesis from: '{}'. Appending {} !".format(question_text, BLANK_STR))
        # Strip space, period and question mark at the end of the question and add a blank
        fitb = re.sub(r"[\.\? ]*$", "", question_text.strip()) + " " + BLANK_STR
    return fitb


# Create a hypothesis statement from the the input fill-in-the-blank statement and answer choice.
def create_hypothesis(fitb: str, choice: str, ans_pos: bool) -> str:

    if ". " + BLANK_STR in fitb or fitb.startswith(BLANK_STR):
        choice = choice[0].upper() + choice[1:]
    else:
        choice = choice.lower()
    # Remove period from the answer choice, if the question doesn't end with the blank
    if not fitb.endswith(BLANK_STR):
        choice = choice.rstrip(".")
    # Some questions already have blanks indicated with 2+ underscores
    if not ans_pos:
        hypothesis = re.sub("__+", choice, fitb)
        return hypothesis
    choice = choice.strip()
    m = re.search("__+", fitb)
    start = m.start()

    length = (len(choice) - 1) if fitb.endswith(BLANK_STR) and choice[-1] in ['.', '?', '!'] else len(choice)
    hypothesis = re.sub("__+", choice, fitb)

    return hypothesis, (start, start + length)


# Identify the wh-word in the question and replace with a blank
def replace_wh_word_with_blank(question_str: str):
    # if "What is the name of the government building that houses the U.S. Congress?" in question_str:
    #     print()
    question_str = question_str.replace("What's", "What is")
    question_str = question_str.replace("whats", "what")
    question_str = question_str.replace("U.S.", "US")
    wh_word_offset_matches = []
    wh_words = ["which", "what", "where", "when", "how", "who", "why"]
    for wh in wh_words:
        # Some Turk-authored SciQ questions end with wh-word
        # E.g. The passing of traits from parents to offspring is done through what?

        if wh == "who" and "people who" in question_str:
            continue

        m = re.search(wh + r"\?[^\.]*[\. ]*$", question_str.lower())
        if m:
            wh_word_offset_matches = [(wh, m.start())]
            break
        else:
            # Otherwise, find the wh-word in the last sentence
            m = re.search(wh + r"[ ,][^\.]*[\. ]*$", question_str.lower())
            if m:
                wh_word_offset_matches.append((wh, m.start()))
            # else:
            #     wh_word_offset_matches.append((wh, question_str.index(wh)))

    # If a wh-word is found
    if len(wh_word_offset_matches):
        # Pick the first wh-word as the word to be replaced with BLANK
        # E.g. Which is most likely needed when describing the change in position of an object?
        wh_word_offset_matches.sort(key=lambda x: x[1])
        wh_word_found = wh_word_offset_matches[0][0]
        wh_word_start_offset = wh_word_offset_matches[0][1]
        # Replace the last question mark with period.
        question_str = re.sub(r"\?$", ".", question_str.strip())
        # Introduce the blank in place of the wh-word
        fitb_question = (question_str[:wh_word_start_offset] + BLANK_STR +
                         question_str[wh_word_start_offset + len(wh_word_found):])
        # Drop "of the following" as it doesn't make sense in the absence of a multiple-choice
        # question. E.g. "Which of the following force ..." -> "___ force ..."
        final = fitb_question.replace(BLANK_STR + " of the following", BLANK_STR)
        final = final.replace(BLANK_STR + " of these", BLANK_STR)
        return final

    elif " them called?" in question_str:
        return question_str.replace(" them called?", " " + BLANK_STR + ".")
    elif " meaning he was not?" in question_str:
        return question_str.replace(" meaning he was not?", " he was not " + BLANK_STR + ".")
    elif " one of these?" in question_str:
        return question_str.replace(" one of these?", " " + BLANK_STR + ".")
    elif re.match(r".*[^\.\?] *$", question_str):
        # If no wh-word is found and the question ends without a period/question, introduce a
        # blank at the end. e.g. The gravitational force exerted by an object depends on its
        return question_str + " " + BLANK_STR
    else:
        # If all else fails, assume "this ?" indicates the blank. Used in Turk-authored questions
        # e.g. Virtually every task performed by living organisms requires this?
        return re.sub(r" this[ \?]", " ___ ", question_str)


# Create the output json dictionary from the input json, premise and hypothesis statement
def create_output_dict(input_json: dict, statement: str, label: bool, ans_pos: bool, pos=None) -> dict:
    if "statements" not in input_json:
        input_json["statements"] = []
    if not ans_pos:
        input_json["statements"].append({"label": label, "statement": statement})
    else:
        input_json["statements"].append({"label": label, "statement": statement, "ans_pos": pos})
    return input_json

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Provide at least two arguments: "
                         "json file with hits, output file name")
    convert_to_entailment(sys.argv[1], sys.argv[2])
