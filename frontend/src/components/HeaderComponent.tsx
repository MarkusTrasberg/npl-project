import React from 'react';

const HeaderComponent = () => {
    return (
        <header className="bg-blue-500 p-4 text-white flex justify-between text-center">
            <h1 className="font-bold text-xl text-center">In-Context Learning Evaluation Tool</h1>
            <nav>
                <ul className="flex space-x-4">
                    <li>
                        {/* <Link to="/link1" className="hover:underline">Link 1</Link> */}
                    </li>
                    <li>
                        {/* <Link to="/link2" className="hover:underline">Link 2</Link> */}
                    </li>
                </ul>
            </nav>
        </header>
    );
};

export default HeaderComponent;
