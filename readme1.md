Implementační dokumentace k 1. úloze do IPP 2024/2025  
Jméno a příjmení: Gleb Litvinchuk  
Login: xlitvi02

## Overview

This parser for the SOL25 language is a solution that verifies source code correctness and produces an XML representation of its Abstract Syntax Tree (AST). The implementation follows a clear separation of concerns by handling lexical analysis, syntactic parsing, and semantic validation in distinct stages.

## Key Components

### Lexical Analysis
- **Purpose:** Tokenizes the source code using regular expressions.
- **Highlights:**  
  - Uses Python's `re` module to recognize comments, keywords, identifiers, integers, and strings.
  - Implements custom error handling with `LexicalError` for invalid characters and escape sequences.

### Syntactic Analysis
- **Purpose:** Constructs the AST via recursive descent parsing.
- **Highlights:**  
  - The `SOL25Parser` class parses classes, methods, blocks, and assignments.
  - Ensures method selectors have the correct format and that block parameter counts match the number of expected arguments.
  - Raises `SyntacticError` with appropriate error codes when structure violations are detected.

### Semantic Analysis
- **Purpose:** Validates the meaning of the code beyond syntax.
- **Highlights:**  
  - Checks for duplicate class/method definitions, undeclared variable usage, and improper inheritance (including circular references).
  - Ensures the presence of a `Main` class with a zero-arity `run` method.
  - Uses `SemanticError` for reporting mismatches in variable usage and inheritance issues.

### XML Generation
- **Purpose:** Produces an XML document representing the AST.
- **Highlights:**  
  - Leverages `xml.etree.ElementTree` to build the XML tree and `xml.dom.minidom` to format it.
  - The output conforms to the specified structure with elements for classes, methods, blocks, parameters, and assignments.
  - Includes the first comment (if available) as the program description.

## Modules and Libraries Used

- **argparse:** Simplifies command-line argument parsing, particularly for the `--help` option.
- **re:** Essential for defining token patterns during lexical analysis.
- **xml.etree.ElementTree & xml.dom.minidom:** Employed to generate and prettify the XML representation of the AST.
- **collections.namedtuple:** Provides an immutable structure for representing tokens.

## Program Flow

1. **Argument Handling:** Validates that only the `--help` flag is accepted.
2. **Lexical Analysis:** Tokenizes the input SOL25 code.
3. **Parsing:** Constructs the AST by recursively parsing classes, methods, and expressions.
4. **Semantic Validation:** Ensures all variables, methods, and inheritance rules conform to the specification.
5. **XML Generation:** Transforms the AST into a well-structured XML document.
6. **Error Handling:** Terminates execution with specific error codes for lexical (21), syntactic (22), semantic (31–35), or internal (99) errors.
