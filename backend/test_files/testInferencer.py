from openicl.icl_inferencer.icl_base_inferencer import BaseInferencer

class TestInferencer(BaseInferencer):
    
	def __init__(self,
	      		 api_name,
			     call_api
				):
	
		super().__init__(None, None, None, None, 1, None,
                         None, None, api_name, None, call_api=False)
