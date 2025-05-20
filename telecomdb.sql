Select * From Customer;
show tables;
Select * From telcomcustomer;
select DISTINCT InternetService From telcomcustomer;
select customerID, max(tenure) as Max_tenure From telcomcustomer GROUP BY customerID ORDER BY Max_tenure DESC;
select DISTINCT PaymentMethod, sum(MonthlyCharges), sum(TotalCharges) From telcomcustomer GROUP BY PaymentMethod;