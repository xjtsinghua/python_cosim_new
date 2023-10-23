ccbf_law=None
clf_value=None
clf_value_2=None
input_u1=None
input_u2=None

class global_value():
    def __init__(self):
        self.ccbf_law=None
    def set_ccbf_law_value(self,value):
        global ccbf_law
        ccbf_law=value

    def get_ccbf_law_value(self):
        global ccbf_law
        return ccbf_law

    def print_global_ccbf_law_value(self):
        global ccbf_law
        print('ccbf_law:',ccbf_law)

    def set_clf_value(self,value):
        global clf_value
        clf_value=value

    def set_clf_value_2(self,value):
        global clf_value_2
        clf_value_2=value

    def set_input_u1(self,value):
        global clf_value_2
        clf_value_2=value

    def set_input_u2(self,value):
        global clf_value_2
        clf_value_2=value

    def get_clf_value(self):
        global clf_value
        return clf_value

    def get_clf_value_2(self):
        global clf_value_2
        return clf_value_2


    def get_input_u1(self):
        global clf_value
        return clf_value

    def get_input_u2(self):
        global clf_value_2
        return clf_value_2
