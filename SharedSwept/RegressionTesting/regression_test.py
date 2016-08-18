import json
import nose.tools as nt

def consistency_test(prob,sch,div,tend):
    sourcepath = os.path.abspath(os.path.dirname(__file__))
    regd = open(os.path.join(sourcepath,"testData.json"),"a+")
    regress_dict = json.load(regd)
    #Now test if the run has a dictionary analog.
    dictout = dict()
    for k in len(data):
        dictout =
        json.dump(data, testsave)
