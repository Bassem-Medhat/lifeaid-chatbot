"""
One-shot keyword enrichment script for processed_data.json.

For each entry whose question and existing keywords suggest a known emergency
topic, this script appends natural-language synonym keywords so the TF-IDF
engine can match informal/colloquial queries (now that chatbot_engine.py uses
keywords in the TF-IDF document text via _build_doc_text).

Safe to re-run: it checks for duplicates before adding anything.
"""

import json
import re

DATA_FILE = 'processed_data.json'


# ─── helpers ─────────────────────────────────────────────────────────────────

def _text(item):
    """Return lowercase combined question + existing keywords for pattern checks."""
    q = item.get('question', '').lower()
    kw = ' '.join(item.get('keywords', [])).lower()
    return q + ' ' + kw


def _matches(item, *phrases):
    """Return True if the combined text contains ANY of the given phrases."""
    t = _text(item)
    return any(p.lower() in t for p in phrases)


def _matches_all(item, *phrases):
    """Return True if the combined text contains ALL of the given phrases."""
    t = _text(item)
    return all(p.lower() in t for p in phrases)


def _add(item, *new_keywords):
    """
    Append new keywords that are not already present.
    Each argument is a single keyword string to add.
    """
    existing_lower = {k.lower().strip() for k in item.get('keywords', [])}
    # also check individual terms within existing multi-term strings
    combined_lower = _text(item)
    added = 0
    for kw in new_keywords:
        kw = kw.strip()
        if not kw:
            continue
        if kw.lower() not in existing_lower and kw.lower() not in combined_lower:
            item.setdefault('keywords', []).append(kw)
            combined_lower += ' ' + kw.lower()
            added += 1
    return added


# ─── topic enrichment rules ───────────────────────────────────────────────────

def enrich(item):
    """Apply all topic-specific keyword additions to a single entry. Returns count added."""
    n = 0

    # ── 1. Cardiac arrest / CPR ──────────────────────────────────────────────
    if _matches(item, 'cardiac arrest', 'chest compressions', 'cpr',
                'fell down not breathing', 'not breathing collapsed'):
        n += _add(item,
            'heart stopped', 'no heartbeat', 'lifeless', 'person collapsed',
            'collapsed not breathing', 'resuscitation', 'push on chest',
            'call 911 heart', 'pump chest', 'how to save life')

    # ── 2. Choking — adult (exclude infant/baby entries) ─────────────────────
    if (_matches(item, 'choking', 'chocking', 'choke', 'airway obstruction',
                 'heimlich', 'abdominal thrust', 'airway blockage')
            and not _matches(item, 'infant', 'baby', 'newborn', 'toddler')):
        n += _add(item,
            'food stuck in throat', 'object stuck in throat', 'gasping for air',
            'cannot speak', 'silent choking', 'face turning red choking',
            'hands on throat', 'airway blocked', 'clutching throat',
            'high pitched breathing noise', 'unable to cough')

    # ── 3. Choking — infant / baby ────────────────────────────────────────────
    if _matches(item, 'infant', 'baby', 'newborn') and \
       _matches(item, 'chok', 'breath', 'airway', 'back blow', 'chest thrust'):
        n += _add(item,
            'baby stopped breathing', 'infant not breathing', 'newborn choking',
            'baby choking emergency', 'small baby airway', 'baby going blue',
            'infant back blows', 'baby chest thrusts', 'child choking under 1')

    # ── 4. Choking — alone / self-heimlich ────────────────────────────────────
    if _matches(item, 'alone') and _matches(item, 'chok', 'choking'):
        n += _add(item,
            'choking by yourself', 'self-heimlich', 'chair back thrust',
            'no one to help choking', 'self abdominal thrust', 'choke alone')

    # ── 5. Severe bleeding ────────────────────────────────────────────────────
    if _matches(item, 'bleeding', 'hemorrhag', 'blood loss', 'arterial'):
        n += _add(item,
            'heavy bleeding', 'blood wont stop', 'gushing blood',
            'spurting blood', 'blood everywhere', 'bleeding profusely',
            'deep cut bleeding', 'wound wont stop bleeding',
            'apply pressure to wound')

    # ── 6. Burns ──────────────────────────────────────────────────────────────
    if _matches(item, 'burn', 'scald', 'thermal injury', 'hot liquid',
                'boiling water'):
        n += _add(item,
            'burnt skin', 'skin on fire', 'cool under running water',
            'red raw skin from heat', 'heat injury skin', 'skin blistering burn',
            'third degree burn', 'first degree burn', 'minor burn treatment',
            'remove clothing burn')

    # ── 7. Blisters ───────────────────────────────────────────────────────────
    if _matches(item, 'blister') and not _matches(item, 'frostbite blister'):
        n += _add(item,
            'fluid bubble on skin', 'bubble under skin',
            'friction blister', 'blister from shoes', 'shoe rubbing blister',
            'blister on heel', 'blister on palm', 'water blister',
            'should I pop blister', 'do not pop blister')

    # ── 8. Fractures / broken bones ───────────────────────────────────────────
    if _matches(item, 'fracture', 'broken bone', 'broken arm', 'broken leg',
                'compound fracture', 'open fracture', 'bone sticking out',
                'bone is broken', 'bone broken'):
        n += _add(item,
            'snapped bone', 'cracked bone', 'broken limb', 'bone injury',
            'fell and broke bone', 'bone looks wrong', 'arm bent wrong',
            'deformed limb', 'how to splint', 'immobilize broken bone',
            'swelling around bone', 'cannot move limb')

    # ── 9. Sprains / strains / twisted ───────────────────────────────────────
    if _matches(item, 'sprain', 'strain', 'twisted ankle', 'rolled ankle',
                'RICE', 'ligament injury'):
        n += _add(item,
            'twisted ankle', 'rolled ankle', 'ankle sprain',
            'swollen ankle', 'ankle injury', 'tripped and twisted',
            'RICE method', 'rest ice compression elevation',
            'sports injury ankle', 'knee sprain', 'wrist sprain')

    # ── 10. Allergic reaction (mild–moderate) ─────────────────────────────────
    if _matches(item, 'allergic reaction', 'allerg', 'hives', 'antihistamine') \
            and not _matches(item, 'anaphylax', 'anaphylactic'):
        n += _add(item,
            'rash from allergy', 'swelling after eating', 'allergy attack',
            'throat tightening allergy', 'skin reaction after contact',
            'itching and swelling', 'allergic rash', 'mild allergy treatment',
            'antihistamine for rash')

    # ── 11. Anaphylaxis / severe allergic ─────────────────────────────────────
    if _matches(item, 'anaphylax', 'anaphylactic', 'epipen', 'epinephrine',
                'severe allergic'):
        n += _add(item,
            'severe allergic reaction', 'allergic collapse',
            'cannot breathe from allergy', 'throat swelling shut',
            'allergic emergency', 'epinephrine injection',
            'allergic shock', 'life-threatening allergy',
            'throat closing from allergy', 'hives and cant breathe')

    # ── 12. Bee / wasp / insect sting ─────────────────────────────────────────
    if _matches(item, 'bee sting', 'wasp sting', 'insect sting', 'stinger',
                'bee venom', 'hornet'):
        n += _add(item,
            'stung by bee', 'stung by wasp', 'bug sting pain',
            'stinger in skin', 'sting swelling', 'insect venom sting',
            'remove bee stinger', 'allergic to bee sting')

    # ── 13. Nosebleed ─────────────────────────────────────────────────────────
    if _matches(item, 'nosebleed', 'nose bleed', 'nose bleeding',
                'epistaxis', 'bloody nose', 'blood from nose'):
        n += _add(item,
            'nose won\'t stop bleeding', 'heavy nosebleed',
            'blood pouring from nose', 'keep nose bleeding',
            'pinch nose stop bleeding', 'lean forward nosebleed',
            'nose bleed first aid', 'bleeding nostril')

    # ── 14. Unconscious / not breathing ──────────────────────────────────────
    if _matches(item, 'unconscious', 'unresponsive', 'not breathing',
                'passed out', 'collapse') and \
       _matches(item, 'airway', 'breathing', 'cpr', 'recovery position'):
        n += _add(item,
            'passed out not breathing', 'collapsed and unresponsive',
            'person on floor not responding', 'no signs of breathing',
            'not waking up after collapsing', 'check airway breathing pulse',
            'recovery position unconscious')

    # ── 15. Head injury / concussion ──────────────────────────────────────────
    if _matches(item, 'head injury', 'concussion', 'skull', 'head trauma',
                'brain injury', 'hit head', 'blow to head', 'fell on head',
                'head wound'):
        n += _add(item,
            'hit in the head', 'blow to head', 'head trauma first aid',
            'fell on head', 'TBI traumatic brain injury',
            'dizzy after head hit', 'nausea after head injury',
            'confused after fall', 'vomiting after head hit',
            'bump on head', 'concussion symptoms')

    # ── 16. Jaw / lower jaw injury ────────────────────────────────────────────
    if _matches(item, 'jaw', 'mandible', 'maxillofacial'):
        n += _add(item,
            'broken jaw', 'jaw fracture', 'lower jaw broken',
            'jaw pain and swelling', 'jaw hit hard', 'jaw injury first aid',
            'face injury jaw')

    # ── 17. Snake bite ────────────────────────────────────────────────────────
    if _matches(item, 'snake', 'venom', 'envenomation', 'antivenin', 'antivenom'):
        n += _add(item,
            'bitten by snake', 'snake bite first aid', 'fang marks',
            'swelling after snake bite', 'venomous snake bite',
            'serpent bite', 'rattlesnake bite', 'cobra bite',
            'snake venom symptoms', 'do not suck venom')

    # ── 18. Poisoning / toxic ingestion ───────────────────────────────────────
    if _matches(item, 'toxic substance', 'poisoning', 'ingested', 'swallowed',
                'decontamination'):
        n += _add(item,
            'swallowed poison', 'drank chemicals', 'ate something toxic',
            'accidental poisoning', 'ingested harmful substance',
            'toxic ingestion', 'call poison control',
            'child ate cleaning product', 'ingested bleach')

    # ── 19. Electric shock ────────────────────────────────────────────────────
    if _matches(item, 'electric shock', 'electrocution', 'electrical source',
                'electrocutat', 'live wire', 'lightning'):
        n += _add(item,
            'touched live wire', 'shocked by electricity',
            'electrical accident', 'lightning struck',
            'zapped by electricity', 'electric injury',
            'power line accident', 'electric shock victim')

    # ── 20. Drowning / water rescue ───────────────────────────────────────────
    if _matches(item, 'drown', 'water rescue', 'near drown', 'pulled from water',
                'secondary drowning', 'aspiration water'):
        n += _add(item,
            'person fell in water', 'pulled from water', 'water rescue',
            'near drowning first aid', 'drowned person',
            'submerged person', 'rescued from water',
            'swimmer in trouble', 'inhaled water')

    # ── 21. Heat stroke / exhaustion ──────────────────────────────────────────
    if _matches(item, 'heat stroke', 'heat exhaustion', 'hyperthermia',
                'heatwave', 'overheating', 'heat emergency', 'evaporative cooling'):
        n += _add(item,
            'overheated person', 'hot and confused',
            'collapse from heat', 'sunstroke', 'sun stroke',
            'overheating symptoms', 'high body temperature emergency',
            'heat emergency first aid', 'cool down overheated person',
            'heat related illness')

    # ── 22. Heat cramps ───────────────────────────────────────────────────────
    if _matches(item, 'heat cramp', 'electrolyte depletion', 'sports drink heat'):
        n += _add(item,
            'muscle cramps from heat', 'hot weather cramps',
            'overheating muscle pain', 'dehydration cramps',
            'working in heat cramps')

    # ── 23. Hypothermia / cold exposure ───────────────────────────────────────
    if _matches(item, 'hypothermia', 'cold water', 'core temperature',
                'rewarming', 'dangerously cold'):
        n += _add(item,
            'very cold person', 'freezing person', 'cold emergency',
            'dangerously cold', 'body temperature dropping',
            'shivering uncontrollably', 'cold injury', 'fell in cold water',
            'exposure to cold', 'hypothermia signs')

    # ── 24. Frostbite ─────────────────────────────────────────────────────────
    if _matches(item, 'frostbite', 'frostnip', 'frozen finger', 'frozen toe',
                'frozen skin'):
        n += _add(item,
            'frozen fingers', 'frozen toes', 'skin turned black from cold',
            'white numb skin cold', 'frost bite first aid',
            'thaw frostbite', 'rewarm frozen skin',
            'cold exposure skin injury')

    # ── 25. Seizures ──────────────────────────────────────────────────────────
    if _matches(item, 'seizure', 'convulsion', 'epilep', 'grand mal',
                'postictal', 'fitting'):
        n += _add(item,
            'person shaking uncontrollably', 'fits', 'epileptic attack',
            'convulsing on floor', 'body jerking uncontrollably',
            'tonic clonic seizure', 'protect during seizure',
            'seizure first aid', 'roll on side seizure',
            'seizure recovery position', 'do not restrain seizure')

    # ── 26. Stroke ────────────────────────────────────────────────────────────
    if _matches(item, 'stroke', 'ischemic', 'hemorrhagic stroke', 'TPA',
                'face drooping', 'facial droop', 'slurred speech',
                'one side weak'):
        n += _add(item,
            'FAST test stroke', 'face drooping one side',
            'arm weakness sudden onset', 'speech suddenly slurred',
            'brain attack', 'face arm speech time',
            'sudden confusion stroke', 'sudden severe headache',
            'trouble walking stroke', 'stroke first aid call 911',
            'CVA cerebrovascular accident')

    # ── 27. Heart attack ──────────────────────────────────────────────────────
    if _matches(item, 'heart attack', 'myocardial infarction',
                'angina', 'chest pain', 'coronary'):
        n += _add(item,
            'heart attack symptoms', 'crushing chest pain',
            'squeezing chest pain', 'left arm pain heart',
            'jaw pain nausea heart', 'nausea with chest pain',
            'heart attack first aid', 'chest pain emergency call 911',
            'chest tightness heart', 'pressure in chest heart',
            'shortness of breath chest pain', 'aspirin for heart attack')

    # ── 28. Asthma ────────────────────────────────────────────────────────────
    if _matches(item, 'asthma', 'bronchospasm', 'albuterol', 'inhaler',
                'wheezing'):
        n += _add(item,
            'asthma attack', 'asthma flare up',
            'can\'t breathe asthma', 'wheeze attack',
            'inhaler not working', 'breathing emergency asthma',
            'asthmatic', 'short of breath asthma',
            'salbutamol ventolin asthma')

    # ── 29. Panic attack ──────────────────────────────────────────────────────
    if _matches(item, 'panic attack', 'hyperventilat', 'anxiety attack'):
        n += _add(item,
            'heart racing with anxiety', 'can\'t breathe from panic',
            'intense fear shaking', 'feel like dying anxiety',
            'panic attack symptoms', 'chest tight anxiety',
            'shaking from fear', 'breathing too fast anxiety')

    # ── 30. Anaphylaxis (separate CRITICAL entry) ────────────────────────────
    if _matches(item, 'anaphylax') and \
       _matches(item, 'epinephrine', 'histamine', 'allergen', 'airway'):
        n += _add(item,
            'anaphylactic shock first aid', 'throat closing anaphylaxis',
            'allergic collapse emergency', 'administer epinephrine',
            'life-threatening allergic reaction', 'anaphylaxis call 911',
            'epipen for anaphylaxis')

    # ── 31. Rabies ────────────────────────────────────────────────────────────
    if _matches(item, 'rabies', 'post-exposure prophylaxis', 'zoonotic',
                'neurotropic'):
        n += _add(item,
            'animal bite rabies risk', 'bitten by dog rabies',
            'bitten by bat', 'rabies prevention',
            'post exposure treatment rabies', 'animal scratch infection',
            'stray dog bite', 'wild animal bite')

    # ── 32. Tetanus ───────────────────────────────────────────────────────────
    if _matches(item, 'tetanus', 'clostridium tetani', 'lockjaw'):
        n += _add(item,
            'rusty nail wound', 'puncture wound tetanus risk',
            'tetanus shot needed', 'lockjaw symptoms',
            'deep wound tetanus', 'nail through skin',
            'tetanus booster overdue')

    # ── 33. Food poisoning ────────────────────────────────────────────────────
    if _matches(item, 'food poisoning', 'gastroenteritis', 'salmonella',
                'e. coli', 'contamination'):
        n += _add(item,
            'ate bad food', 'sick from food', 'vomiting after eating',
            'diarrhea from bad food', 'food contamination illness',
            'food poisoning symptoms', 'stomach cramps after eating',
            'nausea from food')

    # ── 34. Wound / cut dressing ──────────────────────────────────────────────
    if _matches(item, 'dress a wound', 'sterile gauze', 'wound care',
                'antibiotic ointment', 'wound dressing'):
        n += _add(item,
            'how to wrap wound', 'wound bandaging steps',
            'cover a cut', 'sterile dressing wound',
            'antiseptic wound care', 'clean wound cover',
            'bandage application')

    # ── 35. Wound infection ───────────────────────────────────────────────────
    if _matches(item, 'pus', 'infected wound', 'wound infection',
                'red streaks', 'discharge wound'):
        n += _add(item,
            'infected cut', 'wound getting worse',
            'red swollen wound', 'red streaks from wound',
            'fever from infected wound', 'wound smells bad',
            'signs of wound infection')

    # ── 36. Minor cuts / scrapes ──────────────────────────────────────────────
    if _matches(item, 'skinned knee', 'small cut', 'scrape', 'abrasion',
                'minor wound') and \
       not _matches(item, 'deep', 'arterial', 'severe'):
        n += _add(item,
            'minor wound treatment', 'small cut treatment',
            'skin graze', 'shallow cut', 'skin abrasion',
            'clean small wound', 'bandaid cut')

    # ── 37. Puncture wound ────────────────────────────────────────────────────
    if _matches(item, 'puncture', 'puncture wound', 'nail', 'tetanus',
                'hydrogen peroxide'):
        n += _add(item,
            'nail through foot', 'nail through hand',
            'stepped on nail', 'stabbed with sharp object',
            'deep puncture first aid', 'puncture wound treatment')

    # ── 38. Impalement ────────────────────────────────────────────────────────
    if _matches(item, 'impale', 'impaled', 'penetrating trauma',
                'penetrating wound', 'protruding object', 'object in chest'):
        n += _add(item,
            'object stuck in body', 'stabbed with object',
            'something embedded in body', 'do not remove impaled object',
            'penetrating wound first aid', 'impaled object first aid',
            'object sticking out of body')

    # ── 39. Amputation ────────────────────────────────────────────────────────
    if _matches(item, 'amputation', 'amputated', 'severed limb',
                'traumatic amputation', 'lost arm', 'lost leg'):
        n += _add(item,
            'cut off arm', 'cut off leg', 'finger cut off',
            'limb severed', 'body part cut off',
            'tourniquet for amputation', 'preserve amputated part',
            'bleeding from stump', 'reattach limb')

    # ── 40. Crush / trapped ───────────────────────────────────────────────────
    if _matches(item, 'crush syndrome', 'trapped', 'collapsed structure',
                'entrapment'):
        n += _add(item,
            'person trapped under debris', 'crushed by weight',
            'building collapse injury', 'rescue from rubble',
            'crush injury first aid', 'compression injury')

    # ── 41. Chest wound / pneumothorax ───────────────────────────────────────
    if _matches(item, 'pneumothorax', 'chest wound', 'thoracic trauma',
                'occlusive dressing', 'lung injury'):
        n += _add(item,
            'sucking chest wound', 'hole in chest', 'air in chest',
            'lung puncture', 'collapsed lung', 'stabbed in chest',
            'chest injury bleeding air')

    # ── 42. Abdominal wound ───────────────────────────────────────────────────
    if _matches(item, 'evisceration', 'abdominal trauma', 'abdominal cut',
                'peritonitis', 'gut exposed'):
        n += _add(item,
            'organs exposed', 'intestines out', 'stabbed in stomach',
            'deep belly wound', 'abdominal evisceration first aid',
            'keep organs moist', 'do not push organs back')

    # ── 43. Internal bleeding ─────────────────────────────────────────────────
    if _matches(item, 'internal bleed', 'hemoperitoneum', 'internal hemorrh',
                'rigid stomach', 'shock after injury'):
        n += _add(item,
            'bleeding inside body', 'blood in urine injury',
            'blood in stool injury', 'abdominal pain after trauma',
            'signs of internal bleeding', 'shock from bleeding inside',
            'internal injury after accident')

    # ── 44. Splinter ──────────────────────────────────────────────────────────
    if _matches(item, 'splinter', 'foreign body under skin',
                'thorn in skin', 'wood in finger'):
        n += _add(item,
            'wood in skin', 'thorn stuck in finger',
            'splinter removal', 'spike in skin',
            'glass in skin', 'embedded splinter',
            'splinter too deep to remove')

    # ── 45. Ear injury ────────────────────────────────────────────────────────
    if _matches(item, 'eardrum', 'tympanic membrane', 'auditory canal',
                'ear canal', 'something in ear', 'object in ear',
                'insect in ear'):
        n += _add(item,
            'ruptured eardrum', 'something stuck in ear',
            'object in ear canal', 'insect crawled in ear',
            'ear emergency first aid', 'ear pain from object',
            'hearing loss from ear injury')

    # ── 46. Eye injury ────────────────────────────────────────────────────────
    if _matches(item, 'eye', 'ocular', 'corneal', 'globe') and \
       _matches(item, 'injury', 'irritat', 'chemical', 'foreign body',
                'flush', 'irrigation', 'stuck in eye'):
        n += _add(item,
            'something in eye', 'chemical in eye', 'flush eye',
            'eye emergency', 'eye chemical splash', 'rinse eye with water',
            'eye pain first aid', 'scratch on eye',
            'foreign object in eye', 'eye injury first aid')

    # ── 47. Chemical / caustic burn ───────────────────────────────────────────
    if _matches(item, 'chemical burn', 'caustic', 'corrosive',
                'contaminated clothing chemical'):
        n += _add(item,
            'acid burn', 'alkali burn', 'chemical on skin',
            'bleach burn', 'drain cleaner burn',
            'rinse chemical off skin', 'remove contaminated clothing',
            'chemical burn first aid')

    # ── 48. Dental emergency ──────────────────────────────────────────────────
    if _matches(item, 'avulsed tooth', 'dental trauma', 'lost tooth',
                'knocked out tooth', 'tooth fell out'):
        n += _add(item,
            'knocked out tooth first aid', 'tooth came out',
            'broken tooth emergency', 'tooth fell out treatment',
            'lost permanent tooth', 'save knocked out tooth',
            'tooth in milk', 'reimplant tooth')

    # ── 49. Dental pain ───────────────────────────────────────────────────────
    if _matches(item, 'dental pain', 'toothache', 'lost filling',
                'pulpitis', 'abscess', 'jaw throbbing'):
        n += _add(item,
            'tooth pain relief', 'toothache first aid',
            'broken tooth pain', 'dental abscess swelling',
            'tooth sensitivity', 'lost filling pain')

    # ── 50. Diabetic emergency ────────────────────────────────────────────────
    if _matches(item, 'hypoglycemia', 'blood sugar', 'insulin',
                'diabetic', 'glucose', 'hyperglycemia'):
        n += _add(item,
            'diabetic emergency', 'blood sugar too low',
            'diabetic collapsing', 'give sugar to diabetic',
            'diabetic confused unresponsive', 'glucose tablets',
            'insulin shock', 'low sugar attack', 'diabetic hypo')

    # ── 51. Syncope / fainting ────────────────────────────────────────────────
    if _matches(item, 'syncope', 'vasovagal', 'faint', 'fainting',
                'passed out briefly', 'orthostatic'):
        n += _add(item,
            'person fainted', 'fainted and fell', 'blacked out briefly',
            'felt faint then collapsed', 'vasovagal episode',
            'lay person down faint', 'recovery position faint',
            'low blood pressure faint')

    # ── 52. Choking signs / recognition ──────────────────────────────────────
    if _matches(item, 'signs of choking', 'how to know if choking',
                'how to tell if choking', 'choking signs', 'recogni'):
        n += _add(item,
            'how to tell someone is choking', 'choking recognition',
            'signs person needs heimlich', 'cannot speak or cough',
            'chest grab choking sign', 'silent choking sign',
            'blue face choking sign', 'distress sign choking')

    # ── 53. Obese / pregnant choking ─────────────────────────────────────────
    if _matches(item, 'obese', 'pregnant') and _matches(item, 'chok'):
        n += _add(item,
            'chest thrusts pregnant', 'chest thrusts obese',
            'Heimlich for pregnant woman', 'modified heimlich obese',
            'choking pregnant woman', 'choking overweight person')

    # ── 54. First aid kit contents ────────────────────────────────────────────
    if _matches(item, 'first aid kit', 'emergency supplies', 'what to keep'):
        n += _add(item,
            'what belongs in first aid kit', 'emergency kit contents',
            'medical supplies to keep at home', 'bandages gauze antiseptic',
            'basic medical kit', 'first aid box contents',
            'emergency preparedness supplies')

    # ── 55. Dehydration ───────────────────────────────────────────────────────
    if _matches(item, 'dehydration', 'rehydration', 'oral rehydration',
                'ORS', 'electrolyte'):
        n += _add(item,
            'not drinking enough water', 'signs of dehydration',
            'dehydrated person', 'rehydrate fast',
            'oral rehydration salts', 'drink fluids dehydration',
            'dehydration symptoms treatment')

    # ── 56. Spinal injury ─────────────────────────────────────────────────────
    if _matches(item, 'spinal injury', 'spine injury', 'spinal stabiliz',
                'neck pain after accident', 'cant feel legs'):
        n += _add(item,
            'broken neck', 'broken back', 'neck injury after accident',
            'do not move spinal injury', 'spine trauma',
            'paralysis from spinal', 'keep still spinal injury',
            'suspected spinal injury first aid')

    # ── 57. AED use ───────────────────────────────────────────────────────────
    if _matches(item, 'aed', 'defibrillator', 'defibrillation',
                'ventricular fibrillation'):
        n += _add(item,
            'automated external defibrillator', 'how to use AED',
            'shock the heart', 'AED pads placement',
            'shock for cardiac arrest', 'AED voice instructions',
            'turn on AED', 'defibrillator for cardiac arrest')

    # ── 58. Chest compression / CPR technique ────────────────────────────────
    if _matches(item, 'chest compressions', 'rescue breaths', '30:2',
                'basic life support', 'how to do cpr', 'perform cpr',
                'cpr steps', 'cpr technique'):
        n += _add(item,
            'cardiopulmonary resuscitation steps', 'push hard fast chest',
            '30 compressions 2 breaths', 'tilt head chin lift',
            'how to give cpr', 'cpr rate 100-120 per minute',
            'hands only cpr', 'compression only cpr',
            'rescue breathing cpr')

    # ── 59. Hyperventilation ──────────────────────────────────────────────────
    if _matches(item, 'hyperventilat', 'respiratory alkalosis',
                'breathing too fast', 'tingling hands fast breathing'):
        n += _add(item,
            'breathing too fast', 'over breathing', 'dizzy from breathing fast',
            'numb tingling from hyperventilation', 'paper bag breathing',
            'calm breathing technique', 'hyperventilation first aid')

    # ── 60. Blood clot / DVT ──────────────────────────────────────────────────
    if _matches(item, 'deep vein thrombosis', 'DVT', 'pulmonary embolism',
                'blood clot', 'anticoagulant'):
        n += _add(item,
            'blood clot in leg', 'swollen leg blood clot',
            'pain and swelling in calf', 'pulmonary embolism signs',
            'blood clot symptoms', 'DVT first aid',
            'leg clot emergency')

    # ── 61. Appendicitis ──────────────────────────────────────────────────────
    if _matches(item, 'appendicitis', 'appendix', 'mcburney', 'peritonitis'):
        n += _add(item,
            'pain in lower right stomach', 'appendix pain',
            'appendicitis symptoms', 'severe abdominal pain right side',
            'stomach pain that could be appendix', 'appendix emergency')

    # ── 62. Choking — unconscious ─────────────────────────────────────────────
    if _matches(item, 'chocking', 'choking') and \
       _matches(item, 'unconscious', 'unconsciousness'):
        n += _add(item,
            'choking person passed out', 'unconscious after choking',
            'collapsed from choking', 'cpr after choking',
            'airway blocked unconscious')

    # ── 63b. Unconscious / unresponsive (general) ────────────────────────────
    if _matches(item, 'unconscious', 'unresponsive', 'not breathing',
                'passed out', 'collapsed') and \
       not _matches(item, 'water rescue', 'explosion', 'drowning'):
        n += _add(item,
            'person on floor', 'lying on floor', 'collapsed on floor',
            'on the ground unconscious', 'fell and not responding',
            'person not waking up', 'found unconscious',
            'unconscious person first aid', 'not responding to calls',
            'collapsed person first aid')

    # ── 63. Meningitis ────────────────────────────────────────────────────────
    if _matches(item, 'meningitis', 'meninges', 'nuchal rigidity',
                'photophobia rash', 'cerebrospinal'):
        n += _add(item,
            'non-blanching rash', 'glass test rash', 'stiff neck rash fever',
            'meningitis symptoms', 'meningococcal rash',
            'purple spots rash emergency', 'meningitis first aid')

    # ── 64. Opioid overdose ───────────────────────────────────────────────────
    if _matches(item, 'opioid overdose', 'heroin overdose', 'naloxone',
                'pinpoint pupils', 'narcan'):
        n += _add(item,
            'drug overdose unconscious', 'heroin overdose first aid',
            'opioid overdose signs', 'administer naloxone',
            'narcan for overdose', 'overdose blue lips',
            'slow breathing overdose', 'opiate overdose')

    # ── 65. Alcohol poisoning ────────────────────────────────────────────────
    if _matches(item, 'alcohol poisoning', 'signs drunk person',
                'when is drunk an emergency'):
        n += _add(item,
            'drunk person not waking', 'alcohol overdose',
            'passed out from drinking', 'alcohol poisoning signs',
            'vomiting unconscious drunk', 'dangerous level of drinking',
            'alcohol emergency first aid')

    # ── 66. Concussion ───────────────────────────────────────────────────────
    if _matches(item, 'concussion symptoms', 'signs of concussion',
                'head injury symptoms'):
        n += _add(item,
            'hit head concussion', 'concussion from fall',
            'memory loss after head hit', 'dizziness after concussion',
            'headache after head injury', 'concussion first aid',
            'concussion signs to watch')

    # ── 67. EpiPen administration ────────────────────────────────────────────
    if _matches(item, 'epipen', 'how to use an epipen', 'auto-injector'):
        n += _add(item,
            'inject epinephrine', 'epipen into thigh',
            'allergy injection', 'administer epipen',
            'auto-injector allergy emergency',
            'push epipen against leg', 'call 911 after epipen')

    # ── 68. Female / atypical heart attack symptoms ───────────────────────────
    if _matches(item, 'jaw pain', 'nausea tiredness heart',
                'female heart attack', 'woman heart attack'):
        n += _add(item,
            'atypical heart attack symptoms', 'heart attack in women',
            'jaw pain could be heart', 'nausea heart attack sign',
            'unusual heart attack symptoms female',
            'fatigue nausea jaw pain emergency')

    # ── 69. Burn — do not use grease / home remedies ─────────────────────────
    if _matches(item, 'grease on burn', 'toothpaste burn', 'butter burn',
                'home remedy burn'):
        n += _add(item,
            'do not put butter on burn', 'do not use toothpaste burn',
            'no home remedies burn', 'run cool water over burn',
            'burn myth treatment')

    # ── 70. Pregnancy / childbirth emergency ─────────────────────────────────
    if _matches(item, 'childbirth', 'deliver', 'labor', 'giving birth',
                'crowning', 'umbilical cord'):
        n += _add(item,
            'emergency birth', 'baby delivery emergency',
            'premature birth first aid', 'labor pains emergency',
            'waters broken', 'baby crowning', 'umbilical cord care',
            'afterbirth emergency')

    return n


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f'Loading {DATA_FILE}...')
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Loaded {len(data)} entries.')

    total_added = 0
    entries_modified = 0

    for item in data:
        n = enrich(item)
        if n > 0:
            total_added += n
            entries_modified += 1

    print(f'\nEnrichment complete:')
    print(f'  Entries modified : {entries_modified}')
    print(f'  Keywords added   : {total_added}')

    print(f'\nWriting updated {DATA_FILE}...')
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print('Done.')


if __name__ == '__main__':
    main()
