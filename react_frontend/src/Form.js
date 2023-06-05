import "./Form.css"
import Button from "react-bootstrap/esm/Button";
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import Row from 'react-bootstrap/Row';

import RangeSlider from 'react-bootstrap-range-slider';
import React, { useState } from 'react';

import PresentResponse from './Response'
import FormText from "react-bootstrap/esm/FormText";

function GridBasicExample() {

    const [value, setValue] = React.useState(70);
    const [type, setType] = React.useState([])
    const [project, setProject] = useState([]);
    const [PresentResponse] = useState([]);


  return (
    <Form id="MainForm">
      <Row>
        <Form.Text>
        Select model
        </Form.Text>
      </Row>
      <Row>
      <Form.Select>
        <option value="Model1">Model1</option>
        <option value="Model2">Model2</option>
        <option value="Model3">Model3</option>
      </Form.Select>
      </Row>
      <Row>
        <Form.Text>
            Select # of samples to train from
        </Form.Text>
      </Row>
      <Row>
        <Col>
        <Form.Group>
            <RangeSlider
            value={value}
            step={10}
            min={0}
            max={100}
            onChange={e => setValue(e.target.value)}
            />
        </Form.Group>
        </Col>
        <Col xs="3">
          <Form.Control value={value}/>
        </Col>
      </Row>
      <Row>
        <Col>
            <Form.Text>
                Select tasks
            </Form.Text>
        </Col>
      </Row>
      <Row>
        <Col>
            <Form.Check 
                type="switch"
                id="task1-switch"
                label="Task1"
            />
        </Col>
        <Col>
            <Form.Check 
                type="switch"
                id="task2-switch"
                label="Task2"
            />
        </Col>
        <Col>
            <Form.Check 
                type="switch"
                id="task3-switch"
                label="Task3"
            />
        </Col>
      </Row>
      <Row>
        <Button
        type="submit" 
        value="Add Todo"
        onClick={async(event) => {
          event.preventDefault();
          const requestOptions = {
            method: 'POST',
            // mode: 'no-cors',
            // headers: { 'Content-Type': 'application/json'},
            body: JSON.stringify({ title: 'React POST Request Example' })
          };
           const response = await fetch('http://localhost:5000/learn', requestOptions)
           if (response.ok){
            console.log(await response)
            setProject(JSON.stringify(await response.json()))
          }
        //   const response = await fetch("http://localhost:5000/learn", {
        //     method: "POST",
        //     headers: {
        //       'Content-Type' : 'application/json'
        //     },
        //     body: JSON.stringify(to_post)
        //   })
        //   .then(response => response.json())
        //   .then(response => PresentResponse(response))
          
        }}
        >
        </Button>
      </Row>
      <Row>
        <FormText>
          {project}
        </FormText>
      </Row>
    </Form>
  );
}

export default GridBasicExample;