import React from 'react';
import { DataGrid } from '@mui/x-data-grid';


export interface ResultsProps {
    result: {
        accuracy: number;
        questions: string[];
        predictions: string[];
        answers: string[];
    } | null | boolean;
}

function Results( {result}: ResultsProps) {
  const rows = (result && result instanceof Object)
  ?  result.questions.map((question, index) => ({
      id: index,
      question: question,
      prediction: result.predictions[index],
      answer: result.answers[index],
    }))
  : [];


  const columns = [
    { field: 'id', headerName: 'ID', width: 30 },
    { field: 'question', headerName: 'Questions', width: 600, type: 'textarea', whiteSpace: 'normal', wordWrap: 'break-word'},
    { field: 'prediction', headerName: 'Predictions', width: 200 },
    { field: 'answer', headerName: 'Answers', width: 200 },
  ];

  return (
    <div className="ml-10">
      <h1 className="font-bold text-xl w-1/2">Results</h1>
      {result instanceof Object && (
        <>
          <p className="mt-10">Accuracy: {result.accuracy}</p>
          <br></br>
          <div className="">
            <DataGrid
              rows={rows}
              columns={columns}
              initialState={{
                pagination: {
                  paginationModel: { page: 0, pageSize: 5 },
                },
              }}
              pageSizeOptions={[5, 10]}
              className="bg-white"
            />
          </div>
        </>
      )}
      {!result && <p>Press the button to get results.</p>}
    </div>
  );
}

export default Results;