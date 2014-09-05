import preprocess
import entity_extraction
import feature_extraction
import weka


def run_de_ting(start):
    """
    Run the pipeline so far starting at given stage
    """
    if start < 1 or start > 4:
        print 'Only stages between 1 and 4 exist!'
        return 0

    # take required inputs
    if start <= 2:
        no_orgs = input('Should organisations be excluded (1 for yes, 0 for no)? ')

    if start <= 4:
        file_name = raw_input('Enter a name for your WEKA file, .arff will be appended. ')

    # run de ting!
    if start == 1:
        preprocess.preprocessing()
        print 'STAGE ONE FINISHED'

    if start <= 2:
        entity_extraction.extract_all_entities(no_orgs)
        print 'STAGE TWO FINISHED'

    if start <= 3:
        feature_extraction.generate_feature_vectors()
        print 'STAGE THREE FINISHED'

    if start <= 4:
        weka.write_file(file_name)
        print 'STAGE FOUR FINISHED'


if __name__ == '__main__':
    stage = input('What stage of the pipeline would you like to start from? ')
    run_de_ting(stage)
