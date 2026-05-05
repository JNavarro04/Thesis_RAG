def experiment_language():
    # choose between nl, de, en, tr, es, gr, bn, cn
    exp_language = 'en'
    return exp_language


def experiment_gender():
    # choose between 'male' and 'female' -> this requires changes in Unity!
    exp_gender = 'female'
    return exp_gender


def experiment_language_complexity():
    # 0 = normal, 1 = simple
    exp_language_complexity = 0
    return exp_language_complexity


def get_chatgpt_role():
    # role_index 0 = friendly, 1 = grumpy, 2 = bake eggs
    role_index = 0
    language = experiment_language()
    if language == 'nl':
        language_name = "Dutch"
    elif language == "de":
        language_name = "German"
    elif language == "en":
        language_name = "English"
    elif language == "tr":
        language_name = "Turkish"
    elif language == "es":
        language_name = "Spanish"
    elif language == "gr":
        language_name = "Greek"
    elif language == "bn":
        language_name = "Bengali"
    elif language == "cn":
        language_name = "Chinese"

    with open("contextfile.txt") as f:
        context = f.readlines()

    chatgpt_role0 = f"Jij bent een digitale juridische informatiehulp in de stijl van het Juridisch Loket. Geef alleen"\
                    f"algemene, procedurele informatie over voorbereiding op een strafzitting bij de kantonrechter. "\
                    f"Baseer je in eerste instantie op de meegegeven kennisbron in {context}. Wanneer vraagstukken "\
                    f"komen over informatie die niet in deze kennisbron staat, gebruik dan het Large Language model. "\
                    f"Geef geen voorspellingen over schuld, straf of uitkomt. En doe geen definitieve uitspraken. "\
                    f"Geef puur advies en informatie. Geef geen strategisch verdedigingsadvies. Blijf neutraal, "\
                    f"vriendelijk en duidelijk. Praat in eenvoudig Nederlands. Leg kort uit wat het antwoord "\
                    f"praktisch betekent voor de voorbereiding op de zitting."
    chatgpt_role1 = f"You are a grumpy old person who is only interested in the news about the prime minister of Greece Mitsotakis." \
                    f"Your answer contains at maximum three complete sentences. You answer in "\
                    f"the {language_name} language"
    chatgpt_role2 = f"Your are a helpful cooking assistant who is helping someone preparing a cup of tea. "\
                    f"You tell the user to carry out four steps and the user has to confirm each step with OK. "\
                    f"Rules: Talk in the {language_name} language."
    chatgpt_roles = [chatgpt_role0, chatgpt_role1, chatgpt_role2]
    chatgpt_role = chatgpt_roles[role_index]
    return chatgpt_role

