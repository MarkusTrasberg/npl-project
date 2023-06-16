import Button from "react-bootstrap/esm/Button";
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import Row from 'react-bootstrap/Row';
import RangeSlider from 'react-bootstrap-range-slider';
import React, { useState } from 'react';
import axios from 'axios';
import { ResultsProps } from '@/components/Results';


interface Parameters {
  datasets: {
    [key: string]: {
      description: string;
      task: string;
    }};
  inferencers: string[];
  models: string[];
  retrievers: string[];
}

interface GridBasicExampleProps {
  onButtonClick: (result: any) => void; 
}

function GridBasicExample({ onButtonClick }: GridBasicExampleProps) {

    const [dataset_size, setSize] = React.useState(70);
    const [ice_size, setIce] = React.useState(3);
    const [parameters, setParameters] = React.useState<Parameters | null>(null);

    const [model, setModel] = useState("");
    const [inferencer, setInferencer] = useState("");
    const [retriever, setRetriever] = useState("");
    const [datasets, setDatasets] = useState("");

    const handleClick = async (e: React.FormEvent) => { 
      e.preventDefault();
      const dataToSend = { 
        model,  
        inferencer, 
        retriever,
        datasets,
        dataset_size,
        ice_size
      };
      console.log("Sending data.. ", dataToSend)
      try {
        const response = await axios.post('http://localhost:8000/run', dataToSend);
        onButtonClick(response.data);
      } catch (error) {
        console.error('There was an error!', error);
      }
    }

    const fetchParameters = async () => {
      try {
          const response = await axios.get('http://localhost:8000/parameters');
          console.log(response.data);
          setParameters(response.data);
  
          if (response.data) {
            setSize(70);
            setIce(3);
            setModel(response.data.models[0]);
            setInferencer(response.data.inferencers[0]);
            setRetriever(response.data.retrievers[0]);
          }
      } catch (error) {
          console.error('There was an error!', error);
      }
  }
  

  // Call the fetchParameters function when the component mounts
  React.useEffect(() => {
      fetchParameters();
  }, []);
  

  return (
    <div className="text-lg">
      <h1 className="font-bold text-xl"> Training</h1>
    <Form id="MainForm" className="">
      <Row>
        <Form.Text>
        Select model
        </Form.Text>
      </Row>
      <Row>
      <Form.Select onChange={(e) => setModel(e.target.value)}>
      {parameters?.models && parameters.models.map((model, index) => (
                        <option value={model} key={index}>{model}</option>
                    ))}
      </Form.Select>
      </Row>

      <Row>
        <Form.Text>
        Select inferencer
        </Form.Text>
      </Row>
      <Row>
      <Form.Select onChange={(e) => setInferencer(e.target.value)}>
      {parameters?.inferencers && parameters.inferencers.map((inferencer, index) => (
                        <option value={inferencer} key={index}>{inferencer}</option>
                    ))}
      </Form.Select>
      </Row>

      <Row>
        <Form.Text>
        Select retriever
        </Form.Text>
      </Row>
      <Row>
      <Form.Select onChange={(e) => setRetriever(e.target.value)}>
      {parameters?.retrievers && parameters.retrievers.map((retriever, index) => (
                        <option value={retriever} key={index}>{retriever}</option>
                    ))}
      </Form.Select>
      </Row>

      <Row>
        <Form.Text>
            Select # of samples to train from
        </Form.Text>
      </Row>
      
      <Form.Group>
      <Row>

        <Col lg={8} md={8} sm={12} xs={12}>

            <RangeSlider
            value={dataset_size}
            step={10}
            tooltip="off"
            min={0}
            max={100}
            onChange={e => setSize(Number(e.target.value))}
            />
            
        </Col>
        <Col xs="2">
          <Form.Control value={dataset_size}/>
        </Col>
         </Row>
      </Form.Group>
      
      <Form.Group>
      <Form.Text>
            Select ICE size 
        </Form.Text>
      <Row>

        <Col lg={8} md={8} sm={12} xs={12}>

            <RangeSlider
            value={ice_size}
            step={1}
            min={0}
            tooltip="off"
            max={5}
            onChange={e => setIce(Number(e.target.value))}
            />
            
        </Col>
        <Col xs="2">
          <Form.Control value={ice_size}/>
        </Col>
         </Row>
      </Form.Group>
    
      <Row>
    <Col>
        <Form.Text>
            Select task
        </Form.Text>
    </Col>
    </Row>
    {/* <Row>
        {parameters?.datasets && parameters?.datasets.map((dataset: any, index: number) => (
            <Col key={index}>
                <Form.Check 
                    type="switch"
                    id={`${dataset.key}-switch`}
                    label={"   " + dataset.key}
                    onChange={(e) => {
                      if (e.target.checked) {
                          // If checked, add the dataset to the array
                          setDatasets(prevState => [...prevState, dataset]);
                      } else {
                          // If unchecked, remove the dataset from the array
                          setDatasets(prevState => prevState.filter(item => item !== dataset));
                      }
                    }}
                />
            </Col>
        ))}
    </Row> */}

{parameters?.datasets && Object.entries(parameters.datasets).map(([key, dataset], index) => (
    <Row key={index}>
        <Col>
            <Form.Check 
                type="switch"
                id={`${key}-switch`}
                label={<span><b>{key}</b>: {dataset.task} </span> }
                onChange={(e) => {
                  if (e.target.checked) {
                      // If checked, set the key as the selected dataset
                      setDatasets(key);
                  } else {
                      // If unchecked, clear the selected dataset if it's the current one
                      if (datasets === key) {
                          setDatasets("");
                      }
                  }
                }}
                checked={datasets === key} // set the check status based on the current selection
            />
        </Col>
    </Row>
))}




      <Row>
        <Button
        type="submit" 
        value="Add Todo"
        onClick={handleClick}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mt-3"
        >Get results
        </Button>
      </Row>
    </Form>
    </div>

  );
}

export default GridBasicExample;