import React, { useState } from 'react';
interface ResultsProps {
    result: any;
}

function Results( {result}: ResultsProps) {

  return (
    <div className="text-lg text-center">
      <h1 className="font-bold text-xl">Results</h1>
      <p className="mt-10 w-1/2">Accuracy:</p>
      {result}
    </div>
  )}

export default Results;