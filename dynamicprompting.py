from utils import *
import sc_utils
import numpy as np
import fire
from datasets import load_dataset
from copy import deepcopy
from collections import Counter
import time
import datetime
import json
import os
import pickle
import dotenv
from cmath420 import get_answer

ARGS = AttrDict(
    api="openai",
    model="code-davinci-002",
    dataset="aqua",
    split="test",
    tag="",
    size_limit=5000,
    api_time_interval=31,
    api_error_delay=31,
    min_agreement=0.7,  # minimum agreement to accept a prediction
    max_prompt_size=8,
    min_prompt_size=5,
    prompt_mode="hard",
    update_mode="append",
    cot_mode="complexity",  # whether to sample cots for complexity or randomly
    boosting_priority="min",  # how to choose incorrect examples for boosting
    emb_mode="question_and_cot",
    cmath420="level1",
    samples=10,
    sample_len=360,
    iters=10,
    method="few_shot_cot",
    direct_answer_trigger_for_fewshot="The answer is",
    seed=0,
    boosted_prompts=None,
    complexity_prompting=False,
    prompt_tag=None,
)

dotenv.load_dotenv(override=True)

KEYS = (
    [(os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_ORGANIZATION"))]
    + [(os.getenv(f"OPENAI_API_KEY2"), os.getenv(f"OPENAI_ORGANIZATION"))]
)[: int(str(os.getenv("NUM_OPENAI_KEYS")))]

LOG_FILE = None


def write_log(*args, end="\n"):
    line = " ".join([str(a) for a in args])
    print(line, end=end, flush=True)
    global LOG_FILE
    if LOG_FILE is None:
        return
    LOG_FILE.write(line + end)


def prepare_dataset(args, skip_embs=False):
    dataset_string = args.dataset

    if args.complexity_prompting:
        promptfile = f"prompts/{dataset_string}_complex{args.complexity_prompting}.txt"
    elif args.prompt_tag == "nonsense":
        promptfile = f"prompts/nonsense.txt"
    elif args.prompt_tag is not None:
        promptfile = f"prompts/{dataset_string}_{args.prompt_tag}.txt"
    else:
        promptfile = f"prompts/{dataset_string}.txt"
    with open(promptfile, "r") as f:
        INITIAL_FEW_SHOT = f.read()

    if (
        args.prompt_mode == "retrieval"
        and not skip_embs
        and args.get("tr_questions", None) is None
    ):
        split = args.split
        size_limit = args.size_limit
        args.size_limit = 100000
        args.split = "train"
        tr_q, tr_a, _ = prepare_dataset(args, True)
        args.size_limit = size_limit
        args.split = split
        args.tr_questions = np.array(tr_q)
        args.tr_answers = np.array(tr_a)

        assert os.path.exists(f"embeddings/{dataset_string}.npy")
        assert os.path.exists(f"embeddings/{dataset_string}_train.npy")

        a = np.load(f"embeddings/{dataset_string}.npy")[: args.size_limit]
        b = np.load(f"embeddings/{dataset_string}_train.npy")
        args.nearest_matrix = np.flip(a.dot(b.T).argsort(-1), -1)[:, :8]

    elif args.emb_mode == "question_and_cot":
        if os.path.exists(f"embeddings/{dataset_string}_cot.npy"):
            a = np.load(f"embeddings/{dataset_string}_cot.npy")[: args.size_limit]
            args.graham_matrix = a.dot(a.T) * (1.0 - np.eye(len(a)))
    elif args.emb_mode == "question_only":
        if os.path.exists(f"embeddings/{dataset_string}.npy"):
            a = np.load(f"embeddings/{dataset_string}.npy")[: args.size_limit]
            args.graham_matrix = a.dot(a.T) * (1.0 - np.eye(len(a)))
    else:
        raise

    if dataset_string == "gsm8k":
        dataset = load_dataset(dataset_string, "main", split=args.split)
        if args.split == "train":
            rng = np.random.default_rng(seed=0)
            idxs = rng.choice(
                len(dataset), size=min(len(dataset), args.size_limit), replace=False
            )
            dataset = dataset[idxs]
        QUESTIONS = [f"Q: {q}\nA:" for q in dataset["question"]]
        ANSWERS = [
            pred_cleansing(args, a.split("####")[-1]).strip() for a in dataset["answer"]
        ]

    elif dataset_string == "svamp":
        with open("dataset/SVAMP/SVAMP.json", "r") as f:
            data = json.loads(f.read())
        QUESTIONS = [f"Q: {i['Body']} {i['Question']}\nA:" for i in data]
        ANSWERS = [str(float(i["Answer"])) for i in data]

    elif dataset_string == "mmlu570":
        with open(f"dataset/mmlu570/{args.split}.json", "r") as f:
            data = json.loads(f.read())

        QUESTIONS = [i["question"] for i in data[: args.size_limit]]
        ANSWERS = [i["answer"] for i in data[: args.size_limit]]

        prepare_dataset.groups = {}
        for subject in {
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "econometrics",
            "electrical_engineering",
            "elementary_mathematics",
            "formal_logic",
            "global_facts",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
            "human_aging",
            "human_sexuality",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "machine_learning",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "moral_disputes",
            "moral_scenarios",
            "nutrition",
            "philosophy",
            "prehistory",
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
            "virology",
            "world_religions",
        }:
            i = []
            for j, p in enumerate(data[: args.size_limit]):
                if p["type"] == subject:
                    i.append(j)
            prepare_dataset.groups[subject] = i

    elif dataset_string == "cmath420":
        with open("dataset/cmath420/cmath420.json", "r") as f:
            data = json.loads(f.read())
        if args.split == "train":
            dataset = load_dataset("competition_math", split="train")

            candidates = []
            if args.cmath420 == "all":
                for p in dataset:
                    a = get_answer(p["solution"])
                    try:
                        assert "_" not in a
                        assert "\n" not in p["solution"]
                        assert "\n" not in p["problem"]
                        z = float(a)
                        assert z.is_integer()
                    except:
                        continue
                    p["answer"] = get_answer(p["solution"])
                    candidates.append(p)
            elif args.cmath420 == "level1":
                for p in dataset:
                    if p["level"] == "Level 1" and p["type"] == "Prealgebra":
                        a = get_answer(p["solution"])
                        try:
                            assert "_" not in a
                            assert "\n" not in p["solution"]
                            assert "\n" not in p["problem"]
                            z = float(a)
                            assert z.is_integer()
                        except:
                            continue
                        p["answer"] = get_answer(p["solution"])
                        candidates.append(p)
            else:
                raise

            dataset = np.array(candidates)
            rng = np.random.default_rng(seed=0)
            idxs = rng.choice(
                len(dataset), size=min(len(dataset), args.size_limit), replace=False
            )
            data = dataset[idxs]
        QUESTIONS = [f"Q: {i['problem']}\nA:" for i in data][: args.size_limit]
        ANSWERS = [i["answer"] for i in data][: args.size_limit]

        prepare_dataset.groups = {}
        for subject in [
            "Prealgebra",
            "Algebra",
            "Intermediate Algebra",
            "Precalculus",
            "Geometry",
            "Counting & Probability",
            "Number Theory",
        ]:
            for level in ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]:
                i = []
                for j, p in enumerate(data[: args.size_limit]):
                    if p["level"] == level and p["type"] == subject:
                        i.append(j)
                prepare_dataset.groups[(subject, level)] = i

    elif dataset_string == "aqua":
        dataset = load_dataset("aqua_rat", split=args.split)
        if args.split == "train":
            rng = np.random.default_rng(seed=0)
            idxs = rng.choice(
                len(dataset), size=min(len(dataset), args.size_limit), replace=False
            )
            dataset = dataset[idxs]
        amod = (
            lambda a: " ".join(a)
            .replace("A)", "(a) ")
            .replace("B)", "(b) ")
            .replace("C)", "(c) ")
            .replace("D)", "(d) ")
            .replace("E)", "(e) ")
        )
        QUESTIONS = [
            f"Q: {q} Answer Choices: {amod(a)}\nA:"
            for q, a in zip(dataset["question"], dataset["options"])
        ]
        ANSWERS = [a.lower() for a in dataset["correct"]]

    elif dataset_string == "aqua_direct":
        dataset = load_dataset("aqua_rat", split="test")
        amod = (
            lambda a: "{"
            + ", ".join(a)
            .replace("A)", "")
            .replace("B)", "")
            .replace("C)", "")
            .replace("D)", "")
            .replace("E)", "")
            + "}"
        )
        opt = [{o[0]: o[2:] for o in options} for options in dataset["options"]]
        QUESTIONS = [
            f"Q: {q}\nHint: The answer is one of {amod(a)}\nA:"
            for q, a in zip(dataset["question"], dataset["options"])
        ]
        ANSWERS = [o[a] for a, o in zip(dataset["correct"], opt)]

    elif dataset_string == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        QUESTIONS, ANSWERS = data_reader(args)

    elif dataset_string == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        QUESTIONS, ANSWERS = data_reader(args)

    elif dataset_string == "singleq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        QUESTIONS, ANSWERS = data_reader(args)

    elif dataset_string == "mathqa":
        raise

    elif dataset_string == "csqa":
        args.dataset = "commonsensqa"
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        QUESTIONS, ANSWERS = data_reader(args)

    elif dataset_string == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        QUESTIONS, ANSWERS = data_reader(args)

    elif dataset_string == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        QUESTIONS, ANSWERS = data_reader(args)

    elif dataset_string == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        QUESTIONS, ANSWERS = data_reader(args)

    else:
        raise

    return QUESTIONS, ANSWERS, INITIAL_FEW_SHOT


def make_mixed_dataset(args, few_shot=None):
    questions, answers, initial_few_shot = prepare_dataset(args)

    if few_shot is None:
        few_shot = initial_few_shot
    if isinstance(few_shot, str):
        questions = np.array(
            [f"{few_shot}\n\n{q}" for q in questions[: args.size_limit]]
        )
    else:
        questions = np.array(
            [
                f"{few_shot[i]}\n\n{q}"
                for i, q in enumerate(questions[: args.size_limit])
            ]
        )
    answers = np.array(answers)[: args.size_limit]
    return questions, answers, few_shot


import multiprocessing as mp


def pmap(args):
    n, decoder, args, q = args
    time.sleep(np.random.random())
    j = mp.current_process()._identity[0] - 1

    for i in range(5):  # try 5 times, otherwise just return ''
        try:
            if ARGS.prompt_mode == "self_prompted":
                o = decoder.decode(
                    args,
                    q,
                    args.sample_len - i * 40,
                    KEYS[j % len(KEYS)],
                    args.samples,
                    temp=0.7,
                )
            else:
                o = decoder.decode(
                    args,
                    q,
                    args.sample_len,
                    KEYS[j % len(KEYS)],
                    args.samples,
                    temp=0.7,
                    stop=["Q:", "\n\n"],
                )
            write_log(".", end="")

            if n % 5 == 0:
                print(f"*{n}*", end="", flush=True)
            return o
        except openai.error.RateLimitError as e:
            write_log(i, "LIMIT", end=",")
            time.sleep(args.api_error_delay)
            continue
        except Exception as e:
            write_log(str(e)[:30], end=",")
            write_log(f"*KEY*{KEYS[j % len(KEYS)]}*KEY*")
            time.sleep(60)
    return [""]


def pool_get_preds(args, qs, decoder):
    outputs = []
    pool_args = [(i, decoder, args, q) for i, q in enumerate(qs)]

    with mp.Pool(len(KEYS)) as p:
        outputs = list(p.map(pmap, pool_args))

    assert len(outputs) == len(qs)

    return np.array(outputs + [[]], dtype="object")[:-1]  # hack to force numpy object


def get_preds(args, qs, decoder):
    outputs = []
    i = 0
    while len(outputs) < len(qs):
        i += 1
        write_log(".", end="")
        q = qs[len(outputs)]
        try:
            o = decoder.decode(
                args,
                q,
                args.sample_len,
                KEYS[i % len(KEYS)],
                args.samples,
                temp=0.7,
                stop=["Q:", "\n\n"],
            )
            outputs.append(o)
        except openai.error.RateLimitError as e:
            write_log(i, "LIMIT", end=",")
            time.sleep(args.api_error_delay)
            continue
        except Exception as e:
            write_log(e, end=",")
            time.sleep(60)

    return np.array(outputs + [[]], dtype="object")[:-1]  # hack to force numpy object


def filter_dataset(q, a, idxs):
    return q[idxs], a[idxs]


def update_preds(args, preds, new_preds, idxs):
    preds = deepcopy(preds)

    new_confs = evaluate(new_preds, None)[0]

    if args.update_mode == "append":
        for i, q, c in zip(idxs, new_preds, new_confs):
            # if (not args.prompt_mode == 'boosted') and (c >= args.min_agreement):
            #  preds[i] = q # assume LLM knows the answer
            preds[i] = preds[i] + q

        return preds

    elif args.update_mode == "replace":
        confs = evaluate(preds[idxs], None)[0]
        ridxs = idxs[np.where(new_confs > confs)[0]]

        write_log("Replacing:", ridxs)

        for p, q in zip(ridxs, new_preds[np.where(new_confs > confs)[0]]):
            preds[p] = q

        return preds
    else:
        raise


def extract_pred(pred):
    if ARGS.dataset == "aqua":
        p = pred.split("(")[-1].strip().split(")")[0]
        if p not in ["a", "b", "c", "d", "e"]:
            return ""
        return p
    if ARGS.dataset == "aqua_direct":
        return ".".join(pred.split("answer is")[-1].strip().split(".")[:-1])
    elif ARGS.dataset == "cmath420":
        return pred.split("answer is")[-1].strip().split(".")[0]
    elif ARGS.dataset == "mmlu570":
        p = pred.split("(")[-1].strip().split(")")[0]
        if p not in ["a", "b", "c", "d"]:
            return ""
        return p
    elif ARGS.dataset == "svamp":
        e = ".".join(pred.split("answer is")[-1].strip().split(".")[:-1]).replace(
            ".00", ".0"
        )
        try:
            return str(float(e))
        except ValueError:
            return ""
    elif ARGS.dataset == "gsm8k":
        return sc_utils.get_ans(pred)  # can return '' if no answer
    else:
        raise


def extract_preds(preds):
    return list(zip(preds, map(extract_pred, preds)))


def evaluate(preds, answers=None, predict_true_answer=False, weights=None):
    if ARGS.prompt_mode == "self_prompted":
        cots = []
        all_ps = []
        for p in preds:
            extracted = list(zip(*extract_preds(p)))
            cots.append(extracted[0])
            all_ps.append(extracted[1])
        preds = cots
        all_ps = np.array(all_ps, dtype="object")

        if answers is None:
            write_log("Average Prediction Lengths:", np.mean([len(p) for p in all_ps]))

    else:
        all_ps = np.array([list(map(extract_pred, p)) for p in preds], dtype="object")

    if weights is None:
        weights = [np.ones(len(p)) for p in all_ps]

    ps = []
    confs = []
    for i, (p, w) in enumerate(zip(all_ps, weights)):
        if predict_true_answer:
            c = Counter(p)[answers[i]]
            ps.append(answers[i])
            confs.append(float(c) / len(p))
        else:
            c_ = Counter()
            for p_, w_ in zip(p, w):
                c_[
                    p_
                ] += w_  # NOTE: Counter doesn't officially support floats, but it seems to work

            c = list(
                filter(lambda x: x[0], c_.most_common(3))
            )  # filter out non-answers
            if len(c) > 1:
                ps.append(c[0][0])
                confs.append(float(c[0][1]) / len(p))
            elif len(c) == 1:
                ps.append(c[0][0])
                confs.append(float(c[0][1]) / len(p))
            else:
                ps.append("")
                confs.append(1.0)
    ps = np.array(ps)

    if answers is None:
        correct = np.array(-1)
    else:
        correct = ps == np.array(answers)

    idxs = []
    for p, ap in zip(ps, all_ps):
        idxs.append(np.where(np.array(ap) == p)[0])

    # samples a cot from the majority answer
    if ARGS.cot_mode == "complexity":
        len_cots = [
            [len(c.replace("\n", ". ").split(". ")) for c in cot] for cot in preds
        ]
        sample_cots = []
        for cots, i, lc in zip(preds, idxs, len_cots):
            try:
                cots = np.array(cots)[i]
                lc = np.array(lc)[i]
                sample_cots.append(
                    cots[np.random.choice(np.argsort(lc)[-5:])]
                )  # choose one of the longest 5
            except:
                sample_cots.append("")
    elif ARGS.cot_mode == "random":
        sample_cots = [
            (cots[np.random.choice(i)] if len(i) else "")
            for cots, i in zip(preds, idxs)
        ]
    else:
        raise

    confidences = np.array(confs)

    return confidences, correct, np.round(correct.mean(), 3), sample_cots


def get_new_prompt(args, questions, preds, answers):
    if args.prompt_mode == "retrieval":
        res = []
        accs = []
        for j in range(len(questions)):
            nearest_idxs = args.nearest_matrix[j]
            new_few_shots = [
                (f"Q:{q.split('Q:')[-1]}{a}")
                for q, a in zip(
                    args.tr_questions[nearest_idxs], args.tr_answers[nearest_idxs]
                )
            ]
            res.append("\n\n".join(new_few_shots))
        write_log("**************************")
        write_log("PROMPT MODE RETRIEVAL")
        write_log("**************************")

        return res

    min_agreement = args.min_agreement

    correct_answers = np.array(evaluate(preds, answers)[1])
    self_generated_answers = np.array(evaluate(preds, None)[3])

    agreement = evaluate(preds, None)[0]
    i = 1.0
    suitable_agreement = np.logical_and(agreement <= i, agreement >= min_agreement)
    sa_idxs = np.where(suitable_agreement)[0]

    if args.prompt_mode == "nearest_neighbor_suitable":
        res = []
        accs = []
        suitable_graham = args.graham_matrix[:, suitable_agreement]
        nearest_suitable = np.flip(suitable_graham.argsort(-1), -1)[:, : 8 * 3]
        for j in range(len(questions)):
            nearest_idxs = np.random.choice(
                sa_idxs[nearest_suitable[j]],
                min(8, len(nearest_suitable[j])),
                replace=False,
            )
            new_few_shots = [
                (f"Q:{q.split('Q:')[-1]}{a}")
                for q, a in zip(
                    np.array(questions)[nearest_idxs],
                    self_generated_answers[nearest_idxs],
                )
            ]
            accs.append(correct_answers[nearest_idxs].mean())
            res.append("\n\n".join(new_few_shots))
        write_log("Average Prompt Accuracies:", np.mean(accs))
        write_log("**************************")
        return res

    elif args.prompt_mode == "nearest_neighbor_random":
        res = []
        accs = []
        suitable_graham = args.graham_matrix
        nearest = np.flip(suitable_graham.argsort(-1), -1)[:, : 8 * 3]
        for j in range(len(questions)):
            nearest_idxs = np.random.choice(
                nearest[j], min(8, len(nearest[j])), replace=False
            )
            new_few_shots = [
                (f"Q:{q.split('Q:')[-1]}{a}")
                for q, a in zip(
                    np.array(questions)[nearest_idxs],
                    self_generated_answers[nearest_idxs],
                )
            ]
            accs.append(correct_answers[nearest_idxs].mean())
            res.append("\n\n".join(new_few_shots))
        write_log("Average Prompt Accuracies:", np.mean(accs))
        write_log("**************************")
        return res

    elif args.prompt_mode == "self_prompted":
        res = []
        for j in range(len(questions)):
            jpreds = [p for p in preds[j] if "answer is" in p]
            random_idxs = np.random.choice(
                len(jpreds), min(5, len(jpreds)), replace=False
            )
            new_few_shots = [f"A:{jpreds[i]}" for i in random_idxs]
            res.append(
                f"Q:{questions[j].split('Q:')[-1].split('A:')[0]}\n"
                + "\n\n".join(new_few_shots)
            )

        return res

    elif args.prompt_mode in ["self_consistency", "complexity"]:
        return None

    elif args.prompt_mode == "hard":
        suitable_agreement = np.zeros_like(agreement, dtype=bool)
        agreement[agreement < min_agreement] = 10
        suitable_agreement[
            np.argsort(agreement)[: min(24, (agreement >= min_agreement).sum())]
        ] = True

    elif args.prompt_mode == "boosted":
        confs = evaluate(preds, answers, predict_true_answer=True)[0] * np.logical_not(
            correct_answers
        )
        og = (
            confs > 0
        ).sum()  # count of incorrect overall answers with at least 1 correct CoT

        if args.boosting_priority == "random":
            suitable_agreement = confs > 0
        elif args.boosting_priority == "min":
            suitable_agreement = np.zeros_like(confs, dtype=bool)
            confs[confs < 1e-5] = 10
            suitable_agreement[np.argsort(confs)[:16]] = True
        elif args.boosting_priority == "max":
            suitable_agreement = np.zeros_like(confs, dtype=bool)
            suitable_agreement[np.argsort(confs)[-16:]] = True
        elif args.boosting_priority == "bagging":
            suitable_agreement = np.zeros_like(confs, dtype=bool)
            suitable_agreement[
                np.random.choice(len(suitable_agreement), 8, replace=False)
            ] = True
        else:
            raise

        self_generated_answers = np.array(
            evaluate(preds, answers, predict_true_answer=True)[3]
        )
        # but we need to regenerate the self generated answers... .

    else:
        raise NotImplementedError

    idxs = np.random.choice(
        np.where(suitable_agreement)[0], min(8, suitable_agreement.sum()), replace=False
    )
    write_log(f"New prompt size: {len(idxs)} / {suitable_agreement.sum()} ({i})")

    write_log(
        "New prompt accuracy:",
        evaluate(preds[idxs], answers[idxs])[1],
        evaluate(preds[idxs], answers[idxs])[2],
    )
    write_log("Answers:", answers[idxs])
    if args.prompt_mode == "boosted":
        write_log(
            "BOOSTED prompt confidences:",
            evaluate(preds[idxs], answers[idxs], predict_true_answer=True)[2],
        )
    new_few_shots = [
        (f"Q:{q.split('Q:')[-1]}{a}")
        for q, a in zip(np.array(questions)[idxs], self_generated_answers[idxs])
    ]
    new_prompt = "\n\n".join(new_few_shots)
    return new_prompt


def eval_by_group(preds, answers, groups):
    # groups is a list of idx lists
    for i, idxs in enumerate(groups):
        write_log(
            evaluate(np.array(preds)[idxs], np.array(answers)[idxs])[2],
            "-",
            np.round(
                evaluate(np.array(preds)[idxs], np.array(answers)[idxs])[0].mean(), 2
            ),
            f"- {i} ({len(idxs)})",
        )


def main(**kwargs):
    global ARGS

    for k in kwargs:
        ARGS.__dict__[k] = kwargs[k]

    folder = os.path.join("./logs", ARGS.dataset + "_" + ARGS.prompt_mode)
    filename = f"{ARGS.model}_{ARGS.size_limit}_{ARGS.min_agreement}_{ARGS.seed}_{ARGS.update_mode}_{ARGS.tag}"

    if not os.path.exists(folder):
        os.makedirs(folder)

    global LOG_FILE
    LOG_FILE = open(os.path.join(folder, f"{filename}.log"), "a")

    write_log("*****************************")
    write_log(ARGS)
    write_log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    import subprocess, sys

    launch_command = sys.argv[0] + " " + subprocess.list2cmdline(sys.argv[1:])
    write_log(launch_command)
    write_log("Started at:", str(datetime.datetime.now()))
    write_log("*****************************")

    decoder = Decoder(ARGS)
    np.random.seed(ARGS.seed)

    questions, answers, initial_few_shot = make_mixed_dataset(ARGS)
    QUESTIONS = questions
    ANSWERS = answers

    if ARGS.prompt_mode == "retrieval":
        initial_few_shot = get_new_prompt(ARGS, questions, None, answers)

    if ARGS.boosted_prompts is not None:
        with open(ARGS.boosted_prompts, "rb") as f:
            _, _, _, bprompts = pickle.load(f)

    if os.path.exists(os.path.join(folder, f"{filename}.pickle")):
        with open(os.path.join(folder, f"{filename}.pickle"), "rb") as f:
            qs, ps, idxs_, prompts = pickle.load(f)
            preds = ps[-1]

        write_log("LOADED CHECKPOINT", len(ps), len(qs), len(idxs_))
    elif (
        os.path.exists(os.path.join("preds", f"{ARGS.dataset}_{ARGS.seed}.pickle"))
        and (ARGS.split == "test")
        and (ARGS.prompt_mode != "retrieval")
        and not ARGS.complexity_prompting
        and not ARGS.prompt_tag
    ):
        with open(
            os.path.join("preds", f"{ARGS.dataset}_{ARGS.seed}.pickle"), "rb"
        ) as f:
            ps = pickle.load(f)
        assert len(ps[0]) >= len(questions)
        ps = [ps[0][: ARGS.size_limit]]
        preds = ps[0]
        write_log("** [INITIAL PERFORMANCE] Agreement, Actual Performance:")
        a, _, b, _ = evaluate(preds, answers)
        write_log("Overall:", np.mean(a), ",", b)

        prompts = [initial_few_shot]
        qs = [questions]
        idxs_ = [list(range(len(questions)))]
        write_log("LOADED INITIAL PREDS", len(ps), len(qs), len(idxs_))

        new_prompt = get_new_prompt(ARGS, QUESTIONS, ps[0], answers)
        write_log("*****************************")
        if new_prompt is None:
            write_log("Using Self Consistency")
        elif isinstance(new_prompt, str):
            write_log("New prompt:", new_prompt)
        else:
            write_log("Using per-sample prompt... here is a sample:")
            write_log(new_prompt[0])
        write_log("*****************************")
        prompts.append(new_prompt)
    else:
        qs = []
        ps = []
        idxs_ = [list(range(len(questions)))]
        prompts = [initial_few_shot]

    idxs = idxs_[-1]
    not_idxs = None

    k = len(ps)
    if ARGS.boosted_prompts:
        prompts = bprompts

    for i in range(k, k + ARGS.iters):
        write_log("*****************************")
        write_log("ITER:", len(qs))
        write_log("*****************************")
        questions, answers, _ = make_mixed_dataset(ARGS, few_shot=prompts[i])
        if ARGS.boosted_prompts:
            write_log("Using boosted prompts...\n", prompts[i])

        assert np.all(answers == ANSWERS)
        qs.append(questions)

        if len(ps):
            if ARGS.prompt_mode == "boosted" or sum(map(lambda x: len(x[0]), ps)) < 10:
                idxs = idxs_[-1]
            else:
                idxs = np.where(evaluate(preds, answers)[0] < ARGS.min_agreement)[0]
                not_idxs = np.where(evaluate(preds, answers)[0] >= ARGS.min_agreement)[
                    0
                ]
            idxs_.append(idxs)

            questions, _ = filter_dataset(questions, answers, idxs)

        write_log("data size:", len(questions))
        preds = pool_get_preds(ARGS, questions, decoder)

        write_log("** [CURRENT PROMPT] Agreement, Actual Performance:")
        a, _, b, _ = evaluate(preds, answers[idxs])
        write_log("Current:", np.mean(a), ",", b)

        if len(ps):
            preds = update_preds(ARGS, ps[-1], preds, idxs)
        else:
            if ARGS.prompt_mode == "boosted":
                idxs = idxs_[-1]
            else:
                idxs = np.where(evaluate(preds, answers)[0] < ARGS.min_agreement)[0]
                not_idxs = np.where(evaluate(preds, answers)[0] >= ARGS.min_agreement)[
                    0
                ]

        ps.append(preds)

        write_log("** [OVERALL] Agreement, Actual Performance:")
        a, _, b, _ = evaluate(ps[-1][idxs], answers[idxs])
        write_log("Outstanding:", len(idxs), np.mean(a), ",", b)

        if not_idxs is not None:
            a, _, b, _ = evaluate(ps[-1][not_idxs], answers[not_idxs])
            write_log("Solved:", len(not_idxs), np.mean(a), ",", b)
        a, _, b, _ = evaluate(ps[-1], answers)
        write_log("Overall:", np.mean(a), ",", b)

        if ARGS.dataset in ["cmath420", "mmlu570"]:
            for group, i in prepare_dataset.groups.items():
                if i:
                    a, _, b, _ = evaluate(ps[-1][i], answers[i])
                    write_log(f"Group {group}", np.mean(a), ",", b)

        write_log("\n***")
        write_log(len(idxs))
        write_log("***")
        new_prompt = get_new_prompt(ARGS, QUESTIONS, preds, answers)
        write_log("*****************************")
        if new_prompt is None:
            write_log("Using Self Consistency")
        elif isinstance(new_prompt, str):
            write_log("New prompt:", new_prompt)
        else:
            write_log("Using per-sample prompt... here is a sample:")
            write_log(new_prompt[0])
        write_log("*****************************")
        prompts.append(new_prompt)
        if ARGS.boosted_prompts:
            prompts = bprompts

        with open(os.path.join(folder, f"{filename}.pickle"), "wb") as f:
            pickle.dump((qs, ps, idxs_, prompts), f)

    for p in ps:
        # eval_by_type(p, answers, insts)
        write_log(evaluate(p, answers)[2])
        write_log("***")
    write_log("*****************************")
    write_log("Finished at:", str(datetime.datetime.now()))

    LOG_FILE.close()


if __name__ == "__main__":
    """
    See ARGS at top of file for arguments.
    """
    fire.Fire(main)
