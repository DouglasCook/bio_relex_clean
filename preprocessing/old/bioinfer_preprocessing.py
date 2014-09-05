import csv
import os
import xml.etree.cElementTree as ET  # python XML manipulation library, C version because it's way faster!


def remove_extra_crap():
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', '..', 'data/bioinfer/BioInfer_1.2_b.xml'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'bioinfer/bioinfer_1.2.xml'))

    # first parse the tree
    tree = ET.parse(file_in)
    root = tree.getroot()

    # remove ontology tags, don't need them
    rubbish = tree.findall('ontology')
    for r in rubbish:
        root.remove(r)

    # remove all the linkage info, not sure how the format works
    for child in tree.find('sentences'):
        links = child.find('linkages')
        child.remove(links)

    # write stripped version to file
    tree.write(file_out)


def bam():
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'bioinfer/bioinfer_1.2.xml'))

    # first parse the tree
    tree = ET.parse(file_in)

    # print len([c for c in tree.find('sentences')])

    for sentence in tree.find('sentences'):
        # origText gives the sentences
        print sentence.attrib['origText']

        # go through each entity
        for entity in sentence.findall('entity'):
            # if it is an entity of interest ie protein, gene etc
            if entity.attrib['type'] != 'RELATIONSHIP_TEXTBINDING':
                # subtoken ids for words in entity
                ids = [s.attrib['id'] for s in entity]

                tokens = [t.attrib['id'] for t in sentence.findall('token')]
                #subtokens = [s.attrib['id'] for s in t for t in tokens]
                words = [subtoken.attrib['id'] for subtoken in t.findall('subtoken')
                         for t in tokens]
                         #if subtoken.attrib['id'] in ids]

                print entity.attrib['type'], ids


if __name__ == '__main__':
    bam()
