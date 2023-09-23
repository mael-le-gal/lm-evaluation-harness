from lm_eval.tasks.squad import SQuAD2
from mako.template import Template
import os


def read_prompt_file() -> str:
    file = os.environ['SQUAD_OVH_TEMPLATE']
    with open(file) as f:
        return f.read()

def template_until() -> (str, str):
    prompt_file = read_prompt_file()
    split = prompt_file.split("${completion}")
    if len(split) != 2:
        raise Exception("there should be one '${completion}' string in the template (and only one)")
    return split[0], split[1]

class SQuAD2OVH(SQuAD2):
    TEMPLATE, UNTIL = template_until()

    def doc_to_text(self, doc):
        doc['ctx'] = doc.pop('context')
        template = self.TEMPLATE
        prompt = Template(template).render(**doc)
        return prompt

    def until(self):
        return [self.UNTIL]

    def unanswerable(self):
        return "unanswerable"
