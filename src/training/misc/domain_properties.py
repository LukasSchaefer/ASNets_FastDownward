from . import canonicalization

from ...translate import pddl, pddl_parser, translator
from ...translate.pddl import Atom

import os


class DomainProperties(object):
    """
    Manage the analysed properties of a domain.
    """
    def __init__(self, domain, problems=None,
                 fixed_world=None, fixed_objects=None, instantiated_types=None,
                 gnd_static=None, gnd_flexible=None, gnd_problem=None,
                 tpl_pred_static=None, pred_static=None, state_space_size=None,
                 upper_bound_reachable_state_space_size=None,
                 run_analysis=True):

        """
        If no problems are given, the output is undefined.
        :param domain: Domain object (from translate)
        :param problems: List of tuple of PDDL and SAS Problem objects (from translate)
        :prop real_predicates: List of predicate names without "=" predicate
        :param fixed_world: True if for all problems the same objects are used
        :param fixed_objects: Set of objects shared by all problems
        :param instantiated_types: object types instantiated by at least one problem
        :param gnd_static: Set of predicates static for ALL problems (in which
                           they exist)
        :param gnd_flexible: Set of predicates flexible in at least one problem
        :param gnd_problem: List containing for each problem tuple
                            (set of predicates static within problem,
                            set of predicates flexible within problem)
        :param tpl_pred_static: Set of predicate templates which appear only
                                static and never flexible in the problems
        :param pred_static:     Set of predicates which are always static
                                (all its templates are static)
        :param run_analysis: Run analysis after construction
        """
        self.domain = domain
        self.problems = problems
        self.real_predicates = [x for x in self.domain.predicates if x.name != "="]

        self.fixed_world = fixed_world
        self.fixed_objects = fixed_objects
        self.instantiated_types = instantiated_types

        self.gnd_static = gnd_static
        self.gnd_flexible = gnd_flexible
        self.gnd_problem = gnd_problem

        self.tpl_pred_static = tpl_pred_static
        self.pred_static = pred_static

        self.state_space_size = state_space_size
        self.upper_bound_reachable_state_space_size = upper_bound_reachable_state_space_size
        self.analysed = False
        if run_analysis:
            self.analyse()

    def _analyse_fixed_world(self):
        if self.problems is None:
            self.fixed_world = None
        else:
            self.fixed_world = True
            self.fixed_objects = None
            for (pddl, sas) in self.problems:
                new_objects = set(pddl.objects)
                if self.fixed_objects is None:
                    self.fixed_objects = new_objects
                else:
                    if new_objects != self.fixed_objects:
                        self.fixed_world = False
                        self.fixed_objects &= new_objects


    def _analyse_instantiated_types(self):
        self.instantiated_types = set()
        for (pddl, _) in self.problems:
            for obj in pddl.objects:
                self.instantiated_types.add(obj.type_name)

    def _analyse_static_flexible_grounded_predicates_fixed_world(self):
        all_grounded = set(self.problems[0][0].get_grounded_predicates())
        self.gnd_flexible = set()
        self.gnd_problem = []
        some_problem_flexible = set()
        # Get problem static and flexible
        for (pddl_task, sas_task) in self.problems:
            problem_flexible = set()
            for var_names in sas_task.variables.value_names:
                for var_name in var_names:
                    var_atom = Atom.from_string(var_name)
                    problem_flexible.add(var_atom)
            some_problem_flexible.update(problem_flexible)
            problem_static = all_grounded - problem_flexible
            self.gnd_problem.append((problem_static, problem_flexible))

        # Get domain static and flexible
        init_inter = all_grounded.copy()
        init_union = set()
        for (pddl_task, _) in self.problems:
            new_init = set([x for x in pddl_task.init if isinstance(x, Atom) and not x.predicate == "="])
            init_inter &= new_init
            init_union |= new_init
        init_flexible = init_union - init_inter
        self.gnd_flexible = some_problem_flexible | init_flexible
        self.gnd_static = all_grounded - self.gnd_flexible

    def _analyse_static_flexible_grounded_predicates(self):
        if self.fixed_world:
            self._analyse_static_flexible_grounded_predicates_fixed_world()
        else:
            raise NotImplementedError("Analyse static flexible predicates not implemented for not fixed worlds. (e.g. world is variable OR no problems for determining were provided")

    def _analyse_static_predicates_and_templates(self):
        """
        1. Analyse which predicate templates are always static
        2. Analyse which predicate has only static templates = predicate only
           used in a static way.
        :return:
        """
        # Find predicate templates which COULD be always static and to ignore
        predicate_types = {}
        static_predicate_templates = set()
        for gp in self.gnd_static:
            if gp.predicate not in predicate_types:
                args = self.domain.predicate_dict[gp.predicate].arguments
                predicate_types[gp.predicate] = [x.type_name for x in args]
            canonized = canonicalization.canonize_object_lists(
                [gp],
                input_format=canonicalization.Format.ATOM,
                output_format=canonicalization.Format.ATOM,
                types=[predicate_types[gp.predicate]])
            static_predicate_templates.add(canonized[0])

        # Remove predicate templates which are not static
        flexible_predicates = set()
        for gp in self.gnd_flexible:
            if gp.predicate not in predicate_types:
                continue #  predicate not previously encountered, no need to rmv
            flexible_predicates.add(gp.predicate)
            canonized = canonicalization.canonize_object_lists(
                [gp],
                input_format=canonicalization.Format.ATOM,
                output_format=canonicalization.Format.ATOM,
                types=[predicate_types[gp.predicate]])
            static_predicate_templates.discard(canonized[0])
        self.tpl_pred_static = static_predicate_templates
        self.pred_static = predicate_types.keys() - flexible_predicates


    def _analyse_combined_state_space_sizes(self):
        if self.fixed_world:
            self.combined_state_space_size = 2 ** len(self.gnd_flexible)
            self.combined_reachable_state_space_upper_bound = (
                self.problems[0][1].variables.get_state_space_size()
            )
            diff_goals = set()
            for i in range(len(self.problems)):
                diff_goals.add(self.problems[i][0].goal)

            self.combined_reachable_state_space_upper_bound *= len(diff_goals)
            self.combined_state_space_size *= len(diff_goals)



        else:
            # TODO Implement
            pass

    def analyse(self):
        self._analyse_fixed_world()
        if self.problems is not None:
            self._analyse_instantiated_types()
            self._analyse_static_flexible_grounded_predicates()
            self._analyse_static_predicates_and_templates()
            self._analyse_combined_state_space_sizes()
        self.analysed = True

    @staticmethod
    def get_property_for(*paths, **kwargs):
        """
        Analyses the problems in the given paths and returns a DomainProperty
        containing the analysis. Every given path is interpreter as a directory
        in which problem files are searched for. ALL directories are expected to
        contain problems of the SAME domain.
        :param paths: Sequence of directory paths in which all problem files are
                      analysed
        :param paths_problems: Iterable of paths to problem files to analyse
        :param path_domain: Path to the domain file. If not given, then a domain file
                            is searched in the given paths
        :return: DomainProperty containing analysis results
        """
        paths_problems = kwargs.pop("paths_problems", None)
        path_domain = kwargs.pop("path_domain", None)
        if len(paths) == 0 and path_domain is None:
            raise ValueError("No domain file given and no directories to look "
                             "for it.")

        # Find domain file
        if path_domain is None:
            for dir in paths:
                tmp = os.path.join(dir, "domain.pddl")
                if os.path.isfile(tmp):
                    path_domain = tmp
                    break
        if path_domain is None:
            for dir in paths:
                for item in os.listdir(dir):
                    path_item = os.path.join(dir, item)
                    if (item.endswith(".pddl")
                            and item.find("domain") != -1)\
                            and os.path.isfile(path_item):
                        path_domain = path_item
                        break
                if path_domain is not None:
                    break

        # Detect problem files to analyse
        # TODO improve problem file checks
        paths_problems = [] if paths_problems is None else [p for p in paths_problems]
        for dir in paths:
            for item in os.listdir(dir):
                path_item = os.path.join(dir, item)
                if (path_item.endswith(".pddl")
                    and path_item.find("domain") == -1
                    and os.path.isfile(path_item)):
                    paths_problems.append(path_item)


        translator_args = [path_domain, None, "--no-sas-file",
                           "--log-verbosity", "ERROR"]

        (domain_name, domain_requirements, types, type_dict, constants,
         predicates, predicate_dict, functions, actions, axioms) \
            = pddl_parser.parsing_functions.parse_domain_pddl(
            pddl_parser.pddl_file.parse_pddl_file("domain", path_domain))
        domain = pddl.tasks.Domain(domain_name, domain_requirements, types,
                                   constants, predicates, functions, actions,
                                   axioms)


        problems = []
        for path_problem in paths_problems:
            translator_args[1] = path_problem
            pddl_and_sas_task = translator.main(translator_args)
            problems.append(pddl_and_sas_task)
        if len(paths_problems) == 0:
            problems = None
        return DomainProperties(domain, problems)
