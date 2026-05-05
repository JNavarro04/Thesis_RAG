import os
import equipment_config


def run():
    print("Removing existing shelves ...")
    project_path = equipment_config.get_project_path()

    if os.path.isfile(project_path + r'\image_counter_shelve.bak'):
        os.remove(project_path + r'\image_counter_shelve.bak')

    if os.path.isfile(project_path + r'\image_counter_shelve.dat'):
        os.remove(project_path + r'\image_counter_shelve.dat')

    if os.path.isfile(project_path + r'\image_counter_shelve.dir'):
        os.remove(project_path + r'\image_counter_shelve.dir')

    if os.path.isfile(project_path + r'\medicine_counter_shelve.bak'):
        os.remove(project_path + r'\medicine_counter_shelve.bak')

    if os.path.isfile(project_path + r'\medicine_counter_shelve.dat'):
        os.remove(project_path + r'\medicine_counter_shelve.dat')

    if os.path.isfile(project_path + r'\medicine_counter_shelve.dir'):
        os.remove(project_path + r'\medicine_counter_shelve.dir')

    if os.path.isfile(project_path + r'\interlocutors.bak'):
        os.remove(project_path + r'\interlocutors.bak')

    if os.path.isfile(project_path + r'\interlocutors.dat'):
        os.remove(project_path + r'\interlocutors.dat')

    if os.path.isfile(project_path + r'\interlocutors.dir'):
        os.remove(project_path + r'\interlocutors.dir')

    if os.path.isfile(project_path + r'\locations_jane.bak'):
        os.remove(project_path + r'\locations_jane.bak')

    if os.path.isfile(project_path + r'\locations_jane.dat'):
        os.remove(project_path + r'\locations_jane.dat')

    if os.path.isfile(project_path + r'\locations_jane.dir'):
        os.remove(project_path + r'\locations_jane.dir')

    if os.path.isfile(project_path + r'\transcripts_to_action.bak'):
        os.remove(project_path + r'\transcripts_to_action.bak')

    if os.path.isfile(project_path + r'\transcripts_to_action.dat'):
        os.remove(project_path + r'\transcripts_to_action.dat')

    if os.path.isfile(project_path + r'\transcripts_to_action.dir'):
        os.remove(project_path + r'\transcripts_to_action.dir')

    if os.path.isfile(project_path + r'\next_event_shelve.bak'):
        os.remove(project_path + r'\next_event_shelve.bak')

    if os.path.isfile(project_path + r'\next_event_shelve.dat'):
        os.remove(project_path + r'\next_event_shelve.dat')

    if os.path.isfile(project_path + r'\next_event_shelve.dir'):
        os.remove(project_path + r'\next_event_shelve.dir')

    if os.path.isfile(project_path + r'\events_for_today.bak'):
        os.remove(project_path + r'\events_for_today.bak')

    if os.path.isfile(project_path + r'\events_for_today.dat'):
        os.remove(project_path + r'\events_for_today.dat')

    if os.path.isfile(project_path + r'\events_for_today.dir'):
        os.remove(project_path + r'\events_for_today.dir')

    if os.path.isfile(project_path + r'\usecases\events_for_today.bak'):
        os.remove(project_path + r'\usecases\events_for_today.bak')

    if os.path.isfile(project_path + r'\usecases\events_for_today.dat'):
        os.remove(project_path + r'\usecases\events_for_today.dat')

    if os.path.isfile(project_path + r'\usecases\events_for_today.dir'):
        os.remove(project_path + r'\usecases\events_for_today.dir')

    print("Existing shelves removed.")


if __name__ == "__main__":
    run()
