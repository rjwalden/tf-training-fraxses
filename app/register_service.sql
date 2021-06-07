INSERT INTO fraXsesK8S.dbo.dev_evt_tem (srv_cde,end_pnt,pay,inp_sch,com_typ,com_inf) VALUES
('tensorflow','train','{
			"id": @@id@@,
			"obfuscate": @@obfuscate@@,
			"payload":@@payload@@
			}','
	{
	    "requiredInputs": [
	        {
	            "key": "obfuscate",
	            "value": "true",
	            "valueType": "Boolean",
	            "description": "Triggers Obfuscation of Sensitive Portions of the microservice objects for the logs"
	        },
	     
	        {
	            "key": "payload",
	            "value": "",
	            "valueType": "JsonNode",
	            "description": "Json payload to be echoed back"
	        }	
			
	    ],
	    "optionalInputs": [
	
		],
	    "dynamicInputs": {
	        "field": "parameters",
	        "mapping": {}
	    }
	}','REST','{"path":"/tensorflow"}');
