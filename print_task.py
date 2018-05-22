import sys

from src.translate.translator import main as translate
from src.translate.normalize import normalize
from src.translate.instantiate import instantiate, get_fluent_predicates
from src.translate.build_model import compute_model
from src.translate.pddl_to_prolog import translate as pddl_to_prolog

sys.path.append("network_models/asnets")
from problem_meta import ProblemMeta
from asnet_keras_model import ASNet_Model_Builder


def print_grounded_predicates(task_meta):
    print("Grounded Predicates:")
    for prop in task_meta.grounded_predicates:
        prop.dump()
        print("ID: %d" % task_meta.grounded_predicate_name_to_id[prop.__str__()])
        action_name_lists = task_meta.gr_pred_to_related_prop_action_names[prop]
        print("Related Actions: %d" % sum([len(name_list) for name_list in action_name_lists]))
        for action_name_list in action_name_lists:
            print(action_name_list)
        print("Related Action Ids: %s" % str(task_meta.gr_pred_to_related_prop_action_ids[prop]))
        print("")
    print("\n")


def print_propositional_actions(task_meta):
    print("Propositional Actions:")
    for action in task_meta.propositional_actions:
        action.dump()
        print("ID: %d" % task_meta.prop_action_name_to_id[action.name])
        print(task_meta.prop_action_to_related_gr_pred_names[action])
        print("Related Proposition Ids: %s" % str(task_meta.prop_action_to_related_gr_pred_ids[action]))
        print("")
    print("\n")


def assert_correct_len_relatedness_of_propositional_actions(task_meta):
    for action in task_meta.task.actions:
        number_of_related_predicates = len(task_meta.action_to_related_pred_names[action.name])
        for prop_act in task_meta.action_name_to_prop_actions[action.name]:
            assert len(task_meta.prop_action_to_related_gr_pred_names[prop_act]) == number_of_related_predicates,\
                    "Number of related propositions of %s does not match the one of its underlying action\
                     schema %s" % (prop_act.name, action.name)


def print_predicates(task_meta):
    print("Predicates:")
    for pred in task_meta.task.predicates:
        print(pred)
        print(task_meta.pred_to_related_action_names[pred])
        groundings = task_meta.predicate_name_to_grounded_predicates[pred.name]
        print("Number of groundings: %d" % len(groundings))
        print("")
    print("\n")


def print_actions(task_meta):
    print("Actions:")
    for action in task_meta.task.actions:
        print(action)
        print(task_meta.action_to_related_pred_names[action])
        groundings = task_meta.action_name_to_prop_actions[action.name]
        print("Number of groundings: %d" % len(groundings))
        print("")
    print("\n")


def print_grounded_predicates_by_id(task_meta):
    grounded_predicate_name_id_list = []
    for grounded_predicate_name, id in task_meta.grounded_predicate_name_to_id.items():
        grounded_predicate_name_id_list.append((id, grounded_predicate_name))

    for id, grounded_predicate_name in sorted(grounded_predicate_name_id_list):
        print("%s: %s" % (id, grounded_predicate_name))


def print_propositional_actions_by_id(task_meta):
    prop_action_name_id_list = []
    for prop_action_name, id in task_meta.prop_action_name_to_id.items():
        prop_action_name_id_list.append((id, prop_action_name))

    for id, prop_action_name in sorted(prop_action_name_id_list):
        print("%s: %s" % (id, prop_action_name))


def main(argv):
    if argv is None or len(argv) != 3:
        print("Please give two arguments!")
        print("Usage: print_task <domain.pddl> <problem.pddl>")
    else:
        pddl_task, sas_task = translate([argv[1],argv[2]])
        normalize(pddl_task)
        prog = pddl_to_prolog(pddl_task)
        model = compute_model(prog)
        _, grounded_predicates, propositional_actions, _, _ = instantiate(pddl_task, model)
        fluent_predicates = get_fluent_predicates(pddl_task, model)

        pddl_task.simplify(fluent_predicates)
        
        task_meta = ProblemMeta(pddl_task, propositional_actions, grounded_predicates)
        assert_correct_len_relatedness_of_propositional_actions

        # print_propositional_actions(task_meta)
        # print_grounded_predicates(task_meta)

        # print_predicates(task_meta)
        # print_actions(task_meta)

        asnet_builder = ASNet_Model_Builder(task_meta)
        asnet_model = asnet_builder.build_asnet_keras_model(1)
        print(asnet_model.summary())


if __name__ == "__main__":
    main(sys.argv)
